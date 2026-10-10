use super::*;

/// The frames and impulses of a [`SphericalJoint`] within a step.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct SphericalJointSolverData {
    point: PointPart,
    /// The world rotations of the joint frames at the start of the step.
    frames: [Quat; 2],
    /// The axes the swing and the twist were last limited about (the twist's its gradient).
    axes: [Vector; 2],
    swing: LimitPart,
    twist: LimitPart,
    /// The motor's angular impulse.
    motor: Vector,
}

impl JointImpulses for SphericalJointSolverData {
    fn impulses(&self) -> (Vector, AngularVector, f32) {
        let angular = self.swing.along(self.axes[0]) + self.twist.along(self.axes[1]) + self.motor;
        (self.point.impulse, angular, self.motor.length())
    }
}

/// The length of the cone axes' bisector below which the twist is not limited: past some 174
/// degrees of swing. The twist, measured across the shortest swing, is undefined at a half turn
/// and turns ever faster with the swing near it, which its gradient follows; but where it isn't
/// limited it runs on unchecked, and is corrected all at once when it is again, so it is limited
/// as far as it can be (left free from 150 degrees, a shoulder in its 160-degree cone was flung).
const TWIST_UNDEFINED_BISECTOR: f32 = 0.1;

impl SphericalJoint {
    /// The joint frames' axes in world space as the bodies are now: the cone's axis on each
    /// (perpendicular to the [`twist_axis`](Self::twist_axis), which the swing is measured
    /// from and the twist about), and the frames' rotations.
    fn now(
        &self,
        data: &SphericalJointSolverData,
        bodies: &Bodies,
    ) -> ([Vector; 2], [Quat; 2]) {
        let frames = [
            bodies.b1.delta_rotation * data.frames[0],
            bodies.b2.delta_rotation * data.frames[1],
        ];
        let cone = self.twist_axis.any_orthonormal_vector();
        ([frames[0] * cone, frames[1] * cone], frames)
    }

    /// The swing's axis and angle, if it has one.
    fn swing(cones: [Vector; 2]) -> Option<(Vector, f32)> {
        let across = cones[0].cross(cones[1]);
        let length = across.length();
        (length > f32::EPSILON).then(|| (across / length, length.atan2(cones[0].dot(cones[1]))))
    }

    /// The twist and its gradient (how fast it turns as the second body turns, the first body's
    /// turn turning it back), if the swing leaves it defined: the turn about the second cone axis
    /// left once the shortest swing has carried the first onto it.
    fn twist(cones: [Vector; 2], frames: [Quat; 2]) -> Option<(Vector, f32)> {
        if (cones[0] + cones[1]).length() <= TWIST_UNDEFINED_BISECTOR {
            return None;
        }
        // The second frame's rotation from the first's, the shortest swing after the twist about
        // the first cone axis: its twist half angle's sine and cosine are its vector part along
        // that axis and its scalar part, each times the swing's half angle's cosine.
        let mut q = frames[1] * frames[0].inverse();
        if q.w < 0.0 {
            q = -q;
        }
        let (v, c) = (q.xyz(), cones[0]);
        let (sin, cos) = (v.dot(c), q.w);
        let length_squared = sin * sin + cos * cos;
        let gradient = (cos * (cos * c + v.cross(c)) + sin * v) / length_squared;
        Some((gradient, 2.0 * sin.atan2(cos)))
    }

    /// Drives the second frame's rotation relative to the first toward the motor's target
    /// rotation and velocity, with no more than its maximum torque.
    fn drive(
        &self,
        bodies: &mut Bodies,
        data: &mut SphericalJointSolverData,
        frames: [Quat; 2],
        pass: &Pass,
    ) {
        let motor = &self.motor;
        if !motor.enabled {
            // A motor turned off no longer pushes, not even from the last substep.
            data.motor = Default::default();
        }
        let i = bodies.i1 + bodies.i2;
        let Some(soft) = motor
            .enabled
            .then(|| motor_softness(motor.motor_model, i.diagonal().element_sum() / 3.0, pass.h))
            .flatten()
        else {
            return;
        };
        // How far the second frame is turned past its target (the shorter way round); its twist
        // left free, the shortest arc from the target's twist axis to the second frame's.
        let target = frames[0] * motor.target_rotation;
        let twist_axis = frames[1] * self.twist_axis;
        let mut past = if motor.free_twist {
            Quat::from_rotation_arc(target * self.twist_axis, twist_axis)
        } else {
            frames[1] * target.inverse()
        };
        if past.w < 0.0 {
            past = -past;
        }
        let (axis, angle) = past.to_axis_angle();
        let error = axis * angle;
        let mut speed = bodies.spin() - frames[0] * motor.target_velocity;
        if motor.free_twist {
            speed = speed.reject_from_normalized(twist_axis);
        }
        let change = -soft.mass_scale * (angular_mass(bodies) * (speed + soft.bias * error))
            - soft.impulse_scale * data.motor;
        let mut new = (data.motor + change).clamp_length_max(motor.max_torque * pass.h);
        if motor.free_twist {
            // Nothing about the twist axis: the spin about it is the body's own.
            new = new.reject_from_normalized(twist_axis);
        }
        bodies.turn(new - data.motor);
        data.motor = new;
    }
}

impl SoftJoint for SphericalJoint {
    type SolverData = SphericalJointSolverData;

    fn prepare(&self, bodies: [&RigidBodyQueryReadOnlyItem; 2], data: &mut Self::SolverData) {
        let (Some(anchor1), Some(anchor2), Some(basis1), Some(basis2)) = (
            self.local_anchor1(),
            self.local_anchor2(),
            self.local_basis1(),
            self.local_basis2(),
        ) else {
            return;
        };
        data.point.prepare(bodies, [anchor1, anchor2]);
        data.frames = [bodies[0].rotation.0 * basis1, bodies[1].rotation.0 * basis2];
    }

    fn warm_start(&self, mut bodies: Bodies, data: &mut Self::SolverData, pass: &Pass) {
        data.point.warm_start(&mut bodies, pass);
        let (cones, frames) = self.now(data, &bodies);
        let swing = Self::swing(cones).map_or(Vector::ZERO, |s| s.0);
        let twist = Self::twist(cones, frames).map_or(Vector::ZERO, |t| t.0);
        bodies.turn((data.swing.along(swing) + data.twist.along(twist) + data.motor) * pass.warm);
    }

    fn solve(&self, mut bodies: Bodies, data: &mut Self::SolverData, pass: &Pass) {
        let (cones, frames) = self.now(data, &bodies);

        // The motor first, the limits and the point after it, which take priority.
        self.drive(&mut bodies, data, frames, pass);

        match (self.swing_limit, Self::swing(cones)) {
            (Some(limit), Some((axis, angle))) => {
                data.axes[0] = axis;
                data.swing
                    .solve(&mut bodies, limit, angle, axis, self.swing_compliance, pass);
            }
            _ => data.swing = default(),
        }
        match (self.twist_limit, Self::twist(cones, frames)) {
            (Some(limit), Some((gradient, angle))) => {
                data.axes[1] = gradient;
                data.twist.solve(
                    &mut bodies,
                    limit,
                    angle,
                    gradient,
                    self.twist_compliance,
                    pass,
                );
            }
            _ => data.twist = default(),
        }

        // The points held together, unless they are free (an infinite compliance: a drive alone).
        if self.point_compliance.is_finite() {
            data.point.solve(&mut bodies, (self.point_compliance, Rotation::IDENTITY, Vector::INFINITY), pass);
        }
    }
}
