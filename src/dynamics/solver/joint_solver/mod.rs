//! Joints solved as soft impulse constraints in the same substep loop as the contacts.
//!
//! Each substep warm starts, solves with a position bias, and relaxes the joints just before the
//! contacts, as `Box2D` v3 does ("Soft Step", Erin Catto's `Solver2D`, 2024): the joints and the
//! contacts answer each other's impulses within the substep, where a position correction made
//! after the contacts moved bodies resting on them and friction, acting only on velocities,
//! never answered it (ragdolls lying still crept and never slept).
//!
//! - A rigid part (a point, an aligned axis, a reached limit) is soft at the joint frequency,
//!   twice the contacts', its bias only in the solve pass and relaxed away after.
//! - A compliant part (a joint's `*_compliance`, the inverse of its stiffness) is a spring of that
//!   stiffness, and a motor a spring of its model, in both passes.
//! - Limits are speculative: short of a limit, its row only keeps the speed that would carry the
//!   joint past it within the substep.
//!
//! The [`FixedJoint`], [`RevoluteJoint`] and `SphericalJoint` are solved here; the
//! [`PrismaticJoint`] and [`DistanceJoint`] still by [XPBD](super::xpbd), after the contacts.

mod fixed;
mod revolute;
#[cfg(feature = "3d")]
mod spherical;

pub use fixed::FixedJointSolverData;
pub use revolute::RevoluteJointSolverData;
#[cfg(feature = "3d")]
pub use spherical::SphericalJointSolverData;

use crate::{
    dynamics::{
        joints::EntityConstraint,
        solver::{
            ContactSoftnessCoefficients, SolverConfig,
            schedule::{SolverSystems, SubstepSolverSystems},
            softness_parameters::SoftnessCoefficients,
            solver_body::{SolverBody, SolverBodyInertia},
        },
    },
    prelude::*,
};
use bevy::{ecs::component::Mutable, prelude::*};
use core::cmp::Ordering;

/// Solves the [`FixedJoint`], [`RevoluteJoint`] and `SphericalJoint` with the contacts.
pub struct JointSolverPlugin;

impl Plugin for JointSolverPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<FixedJoint, FixedJointSolverData>();
        app.register_required_components::<RevoluteJoint, RevoluteJointSolverData>();
        #[cfg(feature = "3d")]
        app.register_required_components::<SphericalJoint, SphericalJointSolverData>();

        app.add_systems(
            PhysicsSchedule,
            (
                prepare::<FixedJoint>,
                prepare::<RevoluteJoint>,
                #[cfg(feature = "3d")]
                prepare::<SphericalJoint>,
            )
                .chain()
                .in_set(SolverSystems::PrepareJoints),
        );
        app.add_systems(
            PhysicsSchedule,
            (
                write_forces::<FixedJoint>,
                write_forces::<RevoluteJoint>,
                #[cfg(feature = "3d")]
                write_forces::<SphericalJoint>,
            )
                .chain()
                // Each joint's forces are its own entity's, whichever solver wrote them.
                .ambiguous_with_all()
                .in_set(SolverSystems::Finalize),
        );

        // Each stage's joints before its contacts, as Box2D orders them.
        use super::plugin::{solve_contacts, warm_start};
        app.add_systems(
            SubstepSchedule,
            (
                (
                    solve::<FixedJoint, Warm>,
                    solve::<RevoluteJoint, Warm>,
                    #[cfg(feature = "3d")]
                    solve::<SphericalJoint, Warm>,
                )
                    .chain()
                    .before(warm_start)
                    .in_set(SubstepSolverSystems::WarmStart),
                (
                    solve::<FixedJoint, Biased>,
                    solve::<RevoluteJoint, Biased>,
                    #[cfg(feature = "3d")]
                    solve::<SphericalJoint, Biased>,
                )
                    .chain()
                    .before(solve_contacts::<true>)
                    .in_set(SubstepSolverSystems::SolveConstraints),
                (
                    solve::<FixedJoint, Relaxed>,
                    solve::<RevoluteJoint, Relaxed>,
                    #[cfg(feature = "3d")]
                    solve::<SphericalJoint, Relaxed>,
                )
                    .chain()
                    .before(solve_contacts::<false>)
                    .in_set(SubstepSolverSystems::Relax),
            ),
        );
    }
}

/// What one substep's pass over the joints does.
#[derive(Clone, Copy, Debug)]
pub struct Pass {
    /// The substep's length (s).
    pub h: Scalar,
    /// The softness of a rigid part: a contact's against a static body, twice as stiff as one
    /// between dynamic bodies, as `Box2D` has its joints, but damped as the contacts are (at
    /// `Box2D`'s twice critical damping, joints held looser than the contacts pressing on them
    /// let a fallen rider's legs end up under its motorcycle).
    pub rigid: SoftnessCoefficients,
    /// Whether the rigid parts correct their error (the solve pass) or only their speed (relax).
    pub use_bias: bool,
    /// The share of the last substep's impulses applied by the warm start.
    pub warm: Scalar,
}

/// A joint solved with the contacts.
pub trait SoftJoint: Component + EntityConstraint<2> {
    /// The joint's frames and impulses within the step.
    type SolverData: Component<Mutability = Mutable> + Default + JointImpulses;

    /// Takes the joint's frames as the bodies are at the start of the step.
    fn prepare(&self, bodies: [&RigidBodyQueryReadOnlyItem; 2], data: &mut Self::SolverData);

    /// Applies the impulses of the last substep again.
    fn warm_start(&self, bodies: Bodies, data: &mut Self::SolverData, pass: &Pass);

    /// Applies the impulses the joint needs now.
    fn solve(&self, bodies: Bodies, data: &mut Self::SolverData, pass: &Pass);
}

/// A joint's last impulses, read as forces.
pub trait JointImpulses {
    /// The last substep's impulses: linear, angular, and the motor's (N·s, N·m·s).
    fn impulses(&self) -> (Vector, AngularVector, Scalar);
}

/// The two bodies of a joint, with their inverse masses (zero for a body that doesn't move).
pub struct Bodies<'a> {
    /// The first body.
    pub b1: &'a mut SolverBody,
    /// The second body.
    pub b2: &'a mut SolverBody,
    /// The first body's inverse mass.
    pub m1: Vector,
    /// The second body's inverse mass.
    pub m2: Vector,
    /// The first body's inverse angular inertia.
    pub i1: SymmetricTensor,
    /// The second body's inverse angular inertia.
    pub i2: SymmetricTensor,
}

impl Bodies<'_> {
    /// Applies an impulse `p` at the anchors `r1` and `r2`, `p` to the second body and `-p` to the first.
    pub fn push(&mut self, r1: Vector, r2: Vector, p: Vector) {
        self.b1.linear_velocity -= p * self.m1;
        self.b1.angular_velocity -= self.i1 * cross(r1, p);
        self.b2.linear_velocity += p * self.m2;
        self.b2.angular_velocity += self.i2 * cross(r2, p);
    }

    /// Applies an angular impulse, to the second body and its opposite to the first.
    pub fn turn(&mut self, p: AngularVector) {
        self.b1.angular_velocity -= self.i1 * p;
        self.b2.angular_velocity += self.i2 * p;
    }

    /// The second body's angular velocity relative to the first's.
    pub fn spin(&self) -> AngularVector {
        self.b2.angular_velocity - self.b1.angular_velocity
    }

    /// The inverse of the effective mass of a turn about `axis` (any length).
    pub fn inv_mass_about(&self, axis: AngularVector) -> Scalar {
        dot(axis, (self.i1 + self.i2) * axis)
    }
}

/// The scalar product of two angular vectors.
#[inline]
pub(crate) fn dot(a: AngularVector, b: AngularVector) -> Scalar {
    #[cfg(feature = "2d")]
    {
        a * b
    }
    #[cfg(feature = "3d")]
    {
        a.dot(b)
    }
}

/// The softness of a spring of stiffness `k` and damping `c` per unit of the effective mass
/// (an acceleration for each unit of error and of speed) over a substep `h`, as Catto's soft
/// constraints have it; `None` for neither.
pub fn spring(k: Scalar, c: Scalar, h: Scalar) -> Option<SoftnessCoefficients> {
    let a = h * (c + h * k);
    (a > 0.0).then(|| SoftnessCoefficients {
        bias: k / (c + h * k),
        mass_scale: a / (1.0 + a),
        impulse_scale: 1.0 / (1.0 + a),
    })
}

/// The softness of a compliant part (compliance the inverse of its stiffness) of inverse
/// effective mass `k`, or the joint's own if it is that stiff or stiffer, or rigid.
pub fn compliant(compliance: Scalar, k: Scalar, pass: &Pass) -> Option<SoftnessCoefficients> {
    if compliance <= 0.0 {
        return None;
    }
    spring(k / compliance, 0.0, pass.h).filter(|s| s.mass_scale < pass.rigid.mass_scale)
}

/// The softness of a motor's model for a part of inverse effective mass `k`.
pub fn motor_softness(model: MotorModel, k: Scalar, h: Scalar) -> Option<SoftnessCoefficients> {
    match model {
        MotorModel::SpringDamper {
            frequency,
            damping_ratio,
        } => {
            let w = TAU * frequency;
            spring(w * w, 2.0 * damping_ratio * w, h)
        }
        MotorModel::AccelerationBased { stiffness, damping } => spring(stiffness, damping, h),
        MotorModel::ForceBased { stiffness, damping } => spring(stiffness * k, damping * k, h),
    }
}

/// The bias, mass scale and impulse scale of a part with error `c` this pass: a spring's in both
/// passes, a rigid part's only in the solve pass.
pub fn softness(
    c: Scalar,
    spring: Option<SoftnessCoefficients>,
    pass: &Pass,
) -> (Scalar, Scalar, Scalar) {
    match spring {
        Some(s) => (s.bias * c, s.mass_scale, s.impulse_scale),
        None if pass.use_bias => (
            pass.rigid.bias * c,
            pass.rigid.mass_scale,
            pass.rigid.impulse_scale,
        ),
        None => (0.0, 1.0, 0.0),
    }
}

/// The new impulse of one side of a limit, `c` how far the joint is short of it (negative past
/// it), `speed` how fast it closes on it backwards (the rate of `c`), `mass` the effective mass:
/// speculative short of it, a spring (if compliant) or rigid past it. Returns the change.
pub fn limit_side(
    impulse: &mut Scalar,
    c: Scalar,
    speed: Scalar,
    mass: Scalar,
    spring: Option<SoftnessCoefficients>,
    pass: &Pass,
) -> Scalar {
    let (bias, mass_scale, impulse_scale) = if c > 0.0 {
        (c / pass.h, 1.0, 0.0)
    } else {
        softness(c, spring, pass)
    };
    let change = -mass * mass_scale * (speed + bias) - impulse_scale * *impulse;
    let new = (*impulse + change).max(0.0);
    let change = new - *impulse;
    *impulse = new;
    change
}

/// How far past its end a compliant limit stretches (rad) before it stops as a rigid one: a
/// joint's tissue is not stretched without end. Unstopped, a limb pressed by the contacts or
/// flung far past its range stored its spring's energy without bound, and once past a half turn
/// from the range's middle the angle read as short of the other end and the spring let go the
/// other way round at once (a dead elbow pressed 1.8 rad past its end flung the arm).
pub const LIMIT_STRETCH: Scalar = 0.5;

/// An angle held within a limit, both sides of it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Debug, PartialEq)]
pub struct LimitPart {
    /// The last impulses of the lower and the upper side (N·m·s).
    pub impulses: [Scalar; 2],
    /// The last impulses of a compliant limit's rigid stops, [`LIMIT_STRETCH`] past its ends.
    pub stops: [Scalar; 2],
}

impl LimitPart {
    /// Holds the angle `angle` (rad) about `axis` within `limit`.
    pub fn solve(
        &mut self,
        bodies: &mut Bodies,
        limit: AngleLimit,
        angle: Scalar,
        axis: AngularVector,
        compliance: Scalar,
        pass: &Pass,
    ) {
        // Read within a half turn of the range's middle, so that an angle pressed past one end
        // is measured as past it.
        let middle = (limit.min + limit.max) * 0.5;
        let angle = middle + (angle - middle + PI).rem_euclid(TAU) - PI;
        let k = bodies.inv_mass_about(axis);
        if k <= Scalar::EPSILON {
            return;
        }
        let spring = compliant(compliance, k, pass);
        let stretch = if spring.is_some() { LIMIT_STRETCH } else { 0.0 };
        for (side, (past, sign)) in [(angle - limit.min, 1.0), (limit.max - angle, -1.0)]
            .into_iter()
            .enumerate()
        {
            let speed = sign * dot(axis, bodies.spin());
            let change = limit_side(&mut self.impulses[side], past, speed, 1.0 / k, spring, pass);
            bodies.turn(axis * (sign * change));
            if spring.is_some() {
                let speed = sign * dot(axis, bodies.spin());
                let change = limit_side(
                    &mut self.stops[side],
                    past + stretch,
                    speed,
                    1.0 / k,
                    None,
                    pass,
                );
                bodies.turn(axis * (sign * change));
            }
        }
    }

    /// The angular impulse along `axis`.
    pub fn along(&self, axis: AngularVector) -> AngularVector {
        axis * self.net()
    }

    /// The impulse toward the upper end.
    pub fn net(&self) -> Scalar {
        self.impulses[0] + self.stops[0] - self.impulses[1] - self.stops[1]
    }
}

/// A point at which two bodies are held together.
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Debug, PartialEq)]
pub struct PointPart {
    /// The anchor from the first body's center of mass, in world space at the start of the step.
    pub r1: Vector,
    /// The anchor from the second body's center of mass, in world space at the start of the step.
    pub r2: Vector,
    /// The second body's center of mass from the first's at the start of the step.
    pub delta_center: Vector,
    /// The last impulse (N·s).
    pub impulse: Vector,
}

impl PointPart {
    /// Takes the anchors as the bodies are at the start of the step.
    pub fn prepare(&mut self, bodies: [&RigidBodyQueryReadOnlyItem; 2], anchors: [Vector; 2]) {
        let [body1, body2] = bodies;
        self.r1 = body1.rotation * (anchors[0] - body1.center_of_mass.0);
        self.r2 = body2.rotation * (anchors[1] - body2.center_of_mass.0);
        self.delta_center = (body2.position.0 - body1.position.0)
            + (body2.rotation * body2.center_of_mass.0 - body1.rotation * body1.center_of_mass.0);
    }

    fn anchors(&self, bodies: &Bodies) -> (Vector, Vector) {
        (
            bodies.b1.delta_rotation * self.r1,
            bodies.b2.delta_rotation * self.r2,
        )
    }

    /// Applies the last impulse again.
    pub fn warm_start(&mut self, bodies: &mut Bodies, pass: &Pass) {
        let (r1, r2) = self.anchors(bodies);
        bodies.push(r1, r2, self.impulse * pass.warm);
    }

    /// Holds the anchors together.
    pub fn solve(&mut self, bodies: &mut Bodies, compliance: Scalar, pass: &Pass) {
        let (r1, r2) = self.anchors(bodies);
        let k = point_inv_mass(bodies, r1, r2);
        let separation =
            (bodies.b2.delta_position - bodies.b1.delta_position) + (r2 - r1) + self.delta_center;
        // A compliant point is a spring as stiff in every direction, its stiffness against
        // the effective mass averaged over them.
        #[cfg(feature = "2d")]
        let mean = (k.x_axis.x + k.y_axis.y) * 0.5;
        #[cfg(feature = "3d")]
        let mean = (k.x_axis.x + k.y_axis.y + k.z_axis.z) / 3.0;
        let spring = compliant(compliance, mean, pass);
        let (bias, mass_scale, impulse_scale) = softness(1.0, spring, pass);
        let velocity = bodies.b2.velocity_at_point(r2) - bodies.b1.velocity_at_point(r1);
        let impulse = -mass_scale * (k.inverse_or_zero() * (velocity + bias * separation))
            - impulse_scale * self.impulse;
        self.impulse += impulse;
        bodies.push(r1, r2, impulse);
    }
}

/// The inverse effective mass of a point constraint with anchors `r1` and `r2`: how fast an
/// impulse at them parts them.
fn point_inv_mass(bodies: &Bodies, r1: Vector, r2: Vector) -> Matrix {
    let m = bodies.m1 + bodies.m2;
    #[cfg(feature = "2d")]
    let parting = |p: Vector| {
        m * p + (bodies.i1 * cross(r1, p)) * r1.perp() + (bodies.i2 * cross(r2, p)) * r2.perp()
    };
    #[cfg(feature = "3d")]
    let parting = |p: Vector| {
        m * p + (bodies.i1 * r1.cross(p)).cross(r1) + (bodies.i2 * r2.cross(p)).cross(r2)
    };
    #[cfg(feature = "2d")]
    return Matrix::from_cols(parting(Vector::X), parting(Vector::Y));
    #[cfg(feature = "3d")]
    Matrix::from_cols(parting(Vector::X), parting(Vector::Y), parting(Vector::Z))
}

/// The inverse of a summed inverse angular inertia on its free axes.
#[cfg(feature = "3d")]
pub fn angular_mass(bodies: &Bodies) -> SymmetricTensor {
    super::plugin::inverse_on_free_axes(bodies.i1 + bodies.i2)
}

/// Calls `f` with a joint's two bodies, a body that isn't solved (static or asleep) or the
/// lesser of two of unequal dominance standing still.
fn with_bodies(
    bodies: &Query<(&mut SolverBody, &SolverBodyInertia), Without<RigidBodyDisabled>>,
    [entity1, entity2]: [Entity; 2],
    f: impl FnOnce(Bodies),
) {
    let mut dummy1 = SolverBody::DUMMY;
    let mut dummy2 = SolverBody::DUMMY;
    let (mut body1, mut inertia1) = (&mut dummy1, &SolverBodyInertia::DUMMY);
    let (mut body2, mut inertia2) = (&mut dummy2, &SolverBodyInertia::DUMMY);
    // SAFETY: a joint's two bodies are distinct, and the joints are solved one at a time.
    if let Ok((body, inertia)) = unsafe { bodies.get_unchecked(entity1) } {
        body1 = body.into_inner();
        inertia1 = inertia;
    }
    if let Ok((body, inertia)) = unsafe { bodies.get_unchecked(entity2) } {
        body2 = body.into_inner();
        inertia2 = inertia;
    }
    let (inertia1, inertia2) = match (inertia1.dominance() - inertia2.dominance()).cmp(&0) {
        Ordering::Greater => (&SolverBodyInertia::DUMMY, inertia2),
        Ordering::Less => (inertia1, &SolverBodyInertia::DUMMY),
        _ => (inertia1, inertia2),
    };
    f(Bodies {
        b1: body1,
        b2: body2,
        m1: inertia1.effective_inv_mass(),
        m2: inertia2.effective_inv_mass(),
        i1: inertia1.effective_inv_angular_inertia(),
        i2: inertia2.effective_inv_angular_inertia(),
    });
}

/// Takes each joint's frames at the start of the step; a disabled joint's impulses are dropped,
/// so that it starts afresh when it is enabled again.
fn prepare<J: SoftJoint>(
    bodies: Query<RigidBodyQueryReadOnly, Without<RigidBodyDisabled>>,
    mut joints: Query<(&J, &mut J::SolverData, Has<JointDisabled>), Without<RigidBody>>,
) {
    for (joint, mut data, disabled) in &mut joints {
        if disabled {
            *data = default();
        } else if let Ok([body1, body2]) = bodies.get_many(joint.entities()) {
            joint.prepare([&body1, &body2], &mut data);
        }
    }
}

/// Which pass over the joints a substep makes.
trait Stage: Send + Sync + 'static {
    const WARM: bool;
    const BIAS: bool;
}
struct Warm;
struct Biased;
struct Relaxed;
impl Stage for Warm {
    const WARM: bool = true;
    const BIAS: bool = false;
}
impl Stage for Biased {
    const WARM: bool = false;
    const BIAS: bool = true;
}
impl Stage for Relaxed {
    const WARM: bool = false;
    const BIAS: bool = false;
}

fn solve<J: SoftJoint, S: Stage>(
    bodies: Query<(&mut SolverBody, &SolverBodyInertia), Without<RigidBodyDisabled>>,
    mut joints: Query<(&J, &mut J::SolverData), (Without<RigidBody>, Without<JointDisabled>)>,
    time: Res<Time>,
    softness: Res<ContactSoftnessCoefficients>,
    config: Res<SolverConfig>,
) {
    let pass = Pass {
        h: time.delta_seconds_adjusted(),
        rigid: softness.non_dynamic,
        use_bias: S::BIAS,
        warm: config.warm_start_coefficient,
    };
    for (joint, mut data) in &mut joints {
        with_bodies(&bodies, joint.entities(), |bodies| {
            if S::WARM {
                joint.warm_start(bodies, &mut data, &pass);
            } else {
                joint.solve(bodies, &mut data, &pass);
            }
        });
    }
}

/// Writes each joint's last impulses as forces.
fn write_forces<J: SoftJoint>(
    mut joints: Query<(&J::SolverData, &mut JointForces), With<J>>,
    time: Res<Time<Substeps>>,
) {
    let h = time.delta_secs_f64() as Scalar;
    for (data, mut forces) in &mut joints {
        let (linear, angular, motor) = data.impulses();
        forces.set_force(linear / h);
        forces.set_torque(angular / h);
        forces.set_motor_force(motor / h);
    }
}
