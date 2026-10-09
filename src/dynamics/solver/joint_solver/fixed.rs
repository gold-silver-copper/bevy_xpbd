use super::*;

/// The frames and impulses of a [`FixedJoint`] within a step.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct FixedJointSolverData {
    point: PointPart,
    /// The second frame's rotation from the first's at the start of the step.
    #[cfg(feature = "2d")]
    rotation: f32,
    #[cfg(feature = "3d")]
    rotation: Quat,
    /// The impulse holding the frames' rotations together.
    angular: AngularVector,
}

impl JointImpulses for FixedJointSolverData {
    fn impulses(&self) -> (Vector, AngularVector, f32) {
        (self.point.impulse, self.angular, 0.0)
    }
}

impl SoftJoint for FixedJoint {
    type SolverData = FixedJointSolverData;

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
            data.rotation = (*body1.rotation * basis1).angle_between(*body2.rotation * basis2);
        }
        #[cfg(feature = "3d")]
        {
            data.rotation = (body2.rotation.0 * basis2) * (body1.rotation.0 * basis1).inverse();
        }
    }

    fn warm_start(&self, mut bodies: Bodies, data: &mut Self::SolverData, pass: &Pass) {
        data.point.warm_start(&mut bodies, pass);
        bodies.turn(data.angular * pass.warm);
    }

    fn solve(&self, mut bodies: Bodies, data: &mut Self::SolverData, pass: &Pass) {
        // The second frame's turn from the first's, which is to be none.
        #[cfg(feature = "2d")]
        let (error, inv_mass, mean) = {
            let k = bodies.i1 + bodies.i2;
            let error = data.rotation
                + bodies
                    .b1
                    .delta_rotation
                    .angle_between(bodies.b2.delta_rotation);
            (error, k.recip_or_zero(), k)
        };
        #[cfg(feature = "3d")]
        let (error, inv_mass, mean) = {
            let mut turn =
                bodies.b2.delta_rotation * data.rotation * bodies.b1.delta_rotation.inverse();
            if turn.w < 0.0 {
                turn = -turn;
            }
            let k = bodies.i1 + bodies.i2;
            (
                2.0 * turn.xyz(),
                angular_mass(&bodies),
                k.diagonal().element_sum() / 3.0,
            )
        };
        let spring = compliant(self.angle_compliance, mean, pass);
        let (bias, mass_scale, impulse_scale) = softness(1.0, spring, pass);
        let impulse = -mass_scale * (inv_mass * (bodies.spin() + bias * error))
            - impulse_scale * data.angular;
        data.angular += impulse;
        bodies.turn(impulse);

        data.point.solve(&mut bodies, self.point_compliance, pass);
    }
}
