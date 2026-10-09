use super::PointConstraintShared;
use crate::{
    dynamics::solver::{
        solver_body::{SolverBody, SolverBodyInertia},
        xpbd::*,
    },
    prelude::*,
};
use bevy::prelude::*;

/// Constraint data required by the XPBD constraint solver for a [`SphericalJoint`].
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Component, Debug, PartialEq)]
pub struct SphericalJointSolverData {
    pub(super) point_constraint: PointConstraintShared,
    pub(super) swing_axis1: Vector,
    pub(super) swing_axis2: Vector,
    pub(super) twist_axis1: Vector,
    pub(super) twist_axis2: Vector,
    pub(super) total_swing_lagrange: Vector,
    pub(super) total_twist_lagrange: Vector,
}

impl XpbdConstraintSolverData for SphericalJointSolverData {
    fn clear_lagrange_multipliers(&mut self) {
        self.point_constraint.clear_lagrange_multipliers();
        self.total_swing_lagrange = Vector::ZERO;
        self.total_twist_lagrange = Vector::ZERO;
    }

    fn total_position_lagrange(&self) -> Vector {
        self.point_constraint.total_position_lagrange()
    }

    fn total_rotation_lagrange(&self) -> AngularVector {
        self.total_swing_lagrange + self.total_twist_lagrange
    }
}

impl XpbdConstraint<2> for SphericalJoint {
    type SolverData = SphericalJointSolverData;

    fn prepare(
        &mut self,
        bodies: [&RigidBodyQueryReadOnlyItem; 2],
        solver_data: &mut SphericalJointSolverData,
    ) {
        let [body1, body2] = bodies;

        let Some(local_anchor1) = self.local_anchor1() else {
            return;
        };
        let Some(local_anchor2) = self.local_anchor2() else {
            return;
        };
        let Some(local_basis1) = self.local_basis1() else {
            return;
        };
        let Some(local_basis2) = self.local_basis2() else {
            return;
        };

        // Compute the rotation matrices since we're performing so many rotations.
        let rot1_mat = Matrix::from_quat(body1.rotation.0);
        let rot2_mat = Matrix::from_quat(body2.rotation.0);

        // Prepare the point-to-point constraint.
        let point_constraint = &mut solver_data.point_constraint;
        point_constraint.world_r1 = rot1_mat * (local_anchor1 - body1.center_of_mass.0);
        point_constraint.world_r2 = rot2_mat * (local_anchor2 - body2.center_of_mass.0);
        point_constraint.center_difference = (body2.position.0 - body1.position.0)
            + (body2.rotation * body2.center_of_mass.0 - body1.rotation * body1.center_of_mass.0);

        // Prepare the base swing and twist axes.
        let swing_axis = self.twist_axis.any_orthonormal_vector();
        solver_data.swing_axis1 = rot1_mat * (local_basis1 * swing_axis);
        solver_data.swing_axis2 = rot2_mat * (local_basis2 * swing_axis);
        solver_data.twist_axis1 = rot1_mat * (local_basis1 * self.twist_axis);
        solver_data.twist_axis2 = rot2_mat * (local_basis2 * self.twist_axis);
    }

    fn solve(
        &mut self,
        bodies: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        solver_data: &mut SphericalJointSolverData,
        dt: Scalar,
    ) {
        let [body1, body2] = bodies;
        let [inertia1, inertia2] = inertias;

        // Align positions
        solver_data.point_constraint.solve(
            [body1, body2],
            [inertia1, inertia2],
            self.point_compliance,
            dt,
        );

        // Apply swing limits
        self.apply_swing_limits(body1, body2, inertia1, inertia2, solver_data, dt);

        // Apply twist limits
        self.apply_twist_limits(body1, body2, inertia1, inertia2, solver_data, dt);
    }
}

/// The length of the swing axes' bisector below which the twist is not limited: past some 150
/// degrees of swing, where a twist reference's projection across the bisector has shrunk to a
/// quarter of its length and the measured twist swings wildly as the swing changes (a joint
/// flung through a half turn was thrown, its twist corrected toward a value that jumped).
const TWIST_UNDEFINED_BISECTOR: Scalar = 0.5;

impl SphericalJoint {
    /// Applies angle limits to limit the relative rotation of the bodies around the `swing_axis`.
    fn apply_swing_limits(
        &self,
        body1: &mut SolverBody,
        body2: &mut SolverBody,
        inertia1: &SolverBodyInertia,
        inertia2: &SolverBodyInertia,
        solver_data: &mut SphericalJointSolverData,
        dt: Scalar,
    ) {
        if let Some(joint_limit) = self.swing_limit {
            let a1 = body1.delta_rotation * solver_data.swing_axis1;
            let a2 = body2.delta_rotation * solver_data.swing_axis2;

            let n = a1.cross(a2);
            let n_magnitude = n.length();

            if n_magnitude <= Scalar::EPSILON {
                return;
            }

            let n = n / n_magnitude;

            if let Some(correction) = joint_limit.compute_correction(n, a1, a2, PI) {
                let inv_inertia1 = inertia1.effective_inv_angular_inertia();
                let inv_inertia2 = inertia2.effective_inv_angular_inertia();

                solver_data.total_swing_lagrange += self.align_orientation(
                    body1,
                    body2,
                    inv_inertia1,
                    inv_inertia2,
                    correction,
                    0.0,
                    self.swing_compliance,
                    dt,
                );
            }
        }
    }

    /// Applies angle limits to limit the relative rotation of the bodies around the `twist_axis`.
    fn apply_twist_limits(
        &self,
        body1: &mut SolverBody,
        body2: &mut SolverBody,
        inertia1: &SolverBodyInertia,
        inertia2: &SolverBodyInertia,
        solver_data: &mut SphericalJointSolverData,
        dt: Scalar,
    ) {
        if let Some(joint_limit) = self.twist_limit {
            let a1 = body1.delta_rotation * solver_data.swing_axis1;
            let a2 = body2.delta_rotation * solver_data.swing_axis2;

            // The twist is the turn about the second body's swing axis left once the shortest
            // swing has carried the first body's swing axis onto it, undefined when the swing is
            // a half turn and ill-conditioned near it. Short of that the limit holds at any
            // swing (a limit that let go at 120 degrees, as the XPBD paper's clamp did, let the
            // twist run on unchecked past it, then corrected all of it at once as the swing came
            // back, out of no stored energy). It is corrected about that axis, which no swing
            // turns: a twist measured about the bisector of the two axes and corrected about it
            // turned the second axis with every correction, the twist limit and the swing limit
            // undoing each other's work and feeding a swinging limb energy.
            if (a1 + a2).length() <= TWIST_UNDEFINED_BISECTOR {
                return;
            }

            let b1 = body1.delta_rotation * solver_data.twist_axis1;
            let b2 = body2.delta_rotation * solver_data.twist_axis2;
            let carried = Quaternion::from_rotation_arc(a1, a2) * b1;
            let twist = carried.cross(b2).dot(a2).atan2(carried.dot(b2));
            let held = twist.clamp(joint_limit.min, joint_limit.max);

            if twist != held {
                let inv_inertia1 = inertia1.effective_inv_angular_inertia();
                let inv_inertia2 = inertia2.effective_inv_angular_inertia();

                solver_data.total_twist_lagrange += self.align_orientation(
                    body1,
                    body2,
                    inv_inertia1,
                    inv_inertia2,
                    a2 * (twist - held),
                    0.0,
                    self.twist_compliance,
                    dt,
                );
            }
        }
    }
}

impl PositionConstraint for SphericalJoint {}

impl AngularConstraint for SphericalJoint {}
