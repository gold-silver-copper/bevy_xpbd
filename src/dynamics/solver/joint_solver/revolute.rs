use super::*;

/// The frames and impulses of a [`RevoluteJoint`] within a step.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct RevoluteJointSolverData {
    point: PointPart,
    /// The second frame's angle from the first's at the start of the step.
    #[cfg(feature = "2d")]
    angle: Scalar,
    /// The hinge axis on each body and a reference across it, in world space at the start of the step.
    #[cfg(feature = "3d")]
    axes: [Vector; 4],
    /// The hinge axis as last solved.
    #[cfg(feature = "3d")]
    axis: Vector,
    /// The impulse keeping the hinge axes aligned, along the two directions across the first
    /// body's hinge axis ([`across`](Self::across)): it turns with the first body, so that as
    /// the hinge turns in the world no part of it comes to lie along the axis, where no row
    /// takes it back (held as a world vector, a hinge turned about by its body's spin kept a
    /// stale push about its own axis, as strong as the elbow's drive, against it).
    #[cfg(feature = "3d")]
    align: Vector2,
    limit: LimitPart,
    /// The motor's impulse.
    motor: Scalar,
}

impl JointImpulses for RevoluteJointSolverData {
    fn impulses(&self) -> (Vector, AngularVector, Scalar) {
        let axial = self.motor + self.limit.net();
        #[cfg(feature = "2d")]
        let angular = axial;
        #[cfg(feature = "3d")]
        let angular = {
            let across = self.axes[2];
            across * self.align.x + self.axes[0].cross(across) * self.align.y + self.axis * axial
        };
        (self.point.impulse, angular, self.motor.abs())
    }
}

impl RevoluteJointSolverData {
    /// The hinge axis (3D) and the second frame's angle about it from the first's.
    fn hinge(&self, bodies: &Bodies) -> (AngularVector, Scalar) {
        #[cfg(feature = "2d")]
        {
            let angle = self.angle
                + bodies
                    .b1
                    .delta_rotation
                    .angle_between(bodies.b2.delta_rotation);
            (1.0, angle)
        }
        #[cfg(feature = "3d")]
        {
            let a1 = bodies.b1.delta_rotation * self.axes[0];
            let b1 = bodies.b1.delta_rotation * self.axes[2];
            let b2 = bodies.b2.delta_rotation * self.axes[3];
            (a1, b1.cross(b2).dot(a1).atan2(b1.dot(b2)))
        }
    }

    /// The two directions across the hinge axis (3D), on the first body: the hinge's reference
    /// across it and the axis's cross product with it.
    #[cfg(feature = "3d")]
    fn across(&self, bodies: &Bodies, axis: Vector) -> [Vector; 2] {
        let b1 = bodies.b1.delta_rotation * self.axes[2];
        [b1, axis.cross(b1)]
    }
}

impl SoftJoint for RevoluteJoint {
    type SolverData = RevoluteJointSolverData;

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
        let [body1, body2] = bodies;
        #[cfg(feature = "2d")]
        {
            data.angle = (*body1.rotation * basis1).angle_between(*body2.rotation * basis2);
        }
        #[cfg(feature = "3d")]
        {
            let across = self.hinge_axis.any_orthonormal_vector();
            data.axes = [
                body1.rotation.0 * basis1 * self.hinge_axis,
                body2.rotation.0 * basis2 * self.hinge_axis,
                body1.rotation.0 * basis1 * across,
                body2.rotation.0 * basis2 * across,
            ];
        }
    }

    fn warm_start(&self, mut bodies: Bodies, data: &mut Self::SolverData, pass: &Pass) {
        data.point.warm_start(&mut bodies, pass);
        let (axis, _) = data.hinge(&bodies);
        let axial = data.motor + data.limit.net();
        #[cfg(feature = "3d")]
        {
            let [x, y] = data.across(&bodies, axis);
            bodies.turn((x * data.align.x + y * data.align.y) * pass.warm);
        }
        bodies.turn(axis * axial * pass.warm);
    }

    fn solve(&self, mut bodies: Bodies, data: &mut Self::SolverData, pass: &Pass) {
        let (axis, angle) = data.hinge(&bodies);
        #[cfg(feature = "3d")]
        {
            data.axis = axis;
        }

        // The motor first, the limits and the hinge after it, which take priority.
        let k = bodies.inv_mass_about(axis);
        let motor = &self.motor;
        if !motor.enabled {
            // A motor turned off no longer pushes, not even from the last substep.
            data.motor = Default::default();
        }
        if let (true, Some(soft)) = (
            motor.enabled && k > Scalar::EPSILON,
            motor_softness(motor.motor_model, k, pass.h),
        ) {
            let error = (angle - motor.target_position + PI).rem_euclid(TAU) - PI;
            let speed = dot(axis, bodies.spin()) - motor.target_velocity;
            let change = -soft.mass_scale * (speed + soft.bias * error) / k
                - soft.impulse_scale * data.motor;
            let most = motor.max_torque * pass.h;
            let new = (data.motor + change).clamp(-most, most);
            bodies.turn(axis * (new - data.motor));
            data.motor = new;
        }

        match self.angle_limit {
            Some(limit) => {
                data.limit
                    .solve(&mut bodies, limit, angle, axis, self.limit_compliance, pass)
            }
            None => data.limit = default(),
        }

        #[cfg(feature = "3d")]
        {
            // The hinge axes held together: the turn across the first's axis.
            let a2 = bodies.b2.delta_rotation * data.axes[1];
            let across = data.across(&bodies, axis);
            let i = bodies.i1 + bodies.i2;
            let k = Matrix2::from_cols(
                Vector2::new(across[0].dot(i * across[0]), across[1].dot(i * across[0])),
                Vector2::new(across[0].dot(i * across[1]), across[1].dot(i * across[1])),
            );
            let spring = compliant(self.align_compliance, (k.x_axis.x + k.y_axis.y) * 0.5, pass);
            let (bias, mass_scale, impulse_scale) = softness(1.0, spring, pass);
            let error = axis.cross(a2);
            let spin = bodies.spin();
            let along = |v: Vector| Vector2::new(across[0].dot(v), across[1].dot(v));
            let impulse = -mass_scale * (k.inverse_or_zero() * (along(spin) + bias * along(error)))
                - impulse_scale * data.align;
            data.align += impulse;
            bodies.turn(across[0] * impulse.x + across[1] * impulse.y);
        }

        data.point.solve(&mut bodies, self.point_compliance, pass);
    }
}
