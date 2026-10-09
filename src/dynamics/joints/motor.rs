use bevy::prelude::*;

/// Determines how the joint motor force/torque is computed.
///
/// Different models offer trade-offs between ease of tuning and physical accuracy.
/// The default is a [`SpringDamper`](MotorModel::SpringDamper) model that provides
/// stable, predictable behavior across different configurations.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Debug, PartialEq)]
pub enum MotorModel {
    /// A spring-damper model using implicit Euler integration.
    ///
    /// Unlike the other models, this is unconditionally stable: the implicit formulation
    /// naturally limits the response as frequency increases, preventing overshoot and
    /// oscillation even with aggressive parameters. This makes it easier to tune than
    /// the other models, which can become unstable with high stiffness values.
    ///
    /// This is the recommended model for most use cases.
    ///
    /// # Parameters
    ///
    /// - `frequency`: The natural frequency of the spring in Hz. Higher values create stiffer springs.
    /// - `damping_ratio`: The damping ratio.
    ///   - 0.0 = no damping (oscillates forever)
    ///   - 1.0 = critically damped (fastest approach without overshoot)
    ///   - \> 1.0 = overdamped (slower approach without overshoot)
    ///   - < 1.0 = underdamped (overshoots and oscillates)
    SpringDamper {
        /// The natural frequency of the spring in Hz.
        frequency: f32,
        /// The damping ratio.
        damping_ratio: f32,
    },

    /// The motor force/torque is computed directly from the stiffness and damping parameters.
    ///
    /// The model can be described by the following formula:
    ///
    /// ```text
    /// force = (stiffness * position_error) + (damping * velocity_error)
    /// ```
    ///
    /// This produces physically accurate forces/torques, but requires careful tuning of the
    /// stiffness and damping parameters based on the masses of the connected bodies.
    /// High stiffness values can cause instability (overshoot, oscillation, or divergence),
    /// so parameters must be chosen appropriately for your timestep and mass configuration.
    ///
    /// # Parameters
    ///
    /// - `stiffness`: The stiffness coefficient for position control. Set to zero for pure velocity control.
    /// - `damping`: The damping coefficient for velocity control.
    ForceBased {
        /// The stiffness coefficient for position control.
        stiffness: f32,
        /// The damping coefficient for velocity control.
        damping: f32,
    },

    /// The motor force/torque is computed based on the acceleration required to reach the target.
    ///
    /// The model can be described by the following formula:
    ///
    /// ```text
    /// acceleration = (stiffness * position_error) + (damping * velocity_error)
    /// ```
    ///
    /// This automatically scales the motor force/torque based on the masses of the bodies,
    /// resulting in consistent behavior across different mass configurations.
    /// It is therefore easier to tune compared to the [`ForceBased`](MotorModel::ForceBased) model,
    /// which requires manual adjustment of stiffness and damping based on mass.
    ///
    /// Note that high stiffness values can still cause instability. For unconditionally
    /// stable behavior, use the [`SpringDamper`](MotorModel::SpringDamper) model instead.
    ///
    /// # Parameters
    ///
    /// - `stiffness`: The stiffness coefficient for position control. Set to zero for pure velocity control.
    /// - `damping`: The damping coefficient for velocity control.
    AccelerationBased {
        /// The stiffness coefficient for position control.
        stiffness: f32,
        /// The damping coefficient for velocity control.
        damping: f32,
    },
}

impl Default for MotorModel {
    /// The default motor model: a critically damped spring-damper with 5 Hz frequency.
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl MotorModel {
    /// The default motor model: a critically damped spring-damper with 5 Hz frequency.
    pub const DEFAULT: Self = Self::SpringDamper {
        frequency: 5.0,
        damping_ratio: 1.0,
    };
}

/// A motor for driving the angular motion of a [`RevoluteJoint`].
///
/// Motors are configured as part of a joint, applying torque to drive
/// the joint towards a target velocity and/or position.
///
/// ```ignore
/// RevoluteJoint::new(entity1, entity2)
///     .with_motor(
///         AngularMotor::new(MotorModel::SpringDamper {
///             frequency: 2.0,
///             damping_ratio: 1.0,
///         })
///         .with_target_position(target_angle)
///     )
/// ```
///
/// [`RevoluteJoint`]: crate::dynamics::joints::revolute::RevoluteJoint
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Debug, PartialEq)]
pub struct AngularMotor {
    /// Whether the motor is enabled.
    pub enabled: bool,
    /// The target angular velocity (rad/s).
    pub target_velocity: f32,
    /// The target angle (rad) for position control.
    pub target_position: f32,
    /// The maximum torque the motor can apply (N·m).
    pub max_torque: f32,
    /// The motor model used for computing the motor torque.
    pub motor_model: MotorModel,
}

impl Default for AngularMotor {
    fn default() -> Self {
        Self::new(MotorModel::DEFAULT)
    }
}

impl AngularMotor {
    /// Creates a new angular motor with the given motor model.
    #[inline]
    pub const fn new(motor_model: MotorModel) -> Self {
        Self {
            enabled: true,
            target_velocity: 0.0,
            target_position: 0.0,
            max_torque: f32::MAX,
            motor_model,
        }
    }

    /// Creates a new disabled angular motor with the given motor model.
    ///
    /// To enable the motor later, use [`set_enabled`](Self::set_enabled).
    #[inline]
    pub const fn new_disabled(motor_model: MotorModel) -> Self {
        Self {
            enabled: false,
            ..Self::new(motor_model)
        }
    }

    /// Enables or disables the motor.
    #[inline]
    pub const fn set_enabled(&mut self, enabled: bool) -> &mut Self {
        self.enabled = enabled;
        self
    }

    /// Sets the target angular velocity in radians per second.
    #[inline]
    pub const fn with_target_velocity(mut self, velocity: f32) -> Self {
        self.target_velocity = velocity;
        self
    }

    /// Sets the target position.
    #[inline]
    pub const fn with_target_position(mut self, target_position: f32) -> Self {
        self.target_position = target_position;
        self
    }

    /// Sets the maximum torque the motor can apply.
    #[inline]
    pub const fn with_max_torque(mut self, max_torque: f32) -> Self {
        self.max_torque = max_torque;
        self
    }

    /// Sets the motor model used for computing the motor torque.
    #[inline]
    pub const fn with_motor_model(mut self, motor_model: MotorModel) -> Self {
        self.motor_model = motor_model;
        self
    }
}

/// A motor for driving the linear motion of a [`PrismaticJoint`].
///
/// Motors are configured as part of a joint, applying force to drive
/// the joint towards a target velocity and/or position.
///
/// # Spring-Damper Model
///
/// For stable position control that behaves consistently across different configurations,
/// use [`MotorModel::SpringDamper`]. This uses implicit Euler integration for
/// unconditional stability.
///
/// ```ignore
/// PrismaticJoint::new(entity1, entity2)
///     .with_motor(
///         LinearMotor::new(MotorModel::SpringDamper {
///             frequency: 2.0,
///             damping_ratio: 1.0,
///         })
///         .with_target_position(target_position)
///     )
/// ```
///
/// [`PrismaticJoint`]: crate::dynamics::joints::prismatic::PrismaticJoint
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Debug, PartialEq)]
pub struct LinearMotor {
    /// Whether the motor is enabled.
    pub enabled: bool,
    /// The target linear velocity (m/s).
    pub target_velocity: f32,
    /// The target position (m) for position control.
    pub target_position: f32,
    /// The maximum force the motor can apply (N).
    pub max_force: f32,
    /// The motor model used for computing the motor force.
    pub motor_model: MotorModel,
}

impl Default for LinearMotor {
    fn default() -> Self {
        Self::new(MotorModel::DEFAULT)
    }
}

impl LinearMotor {
    /// Creates a new linear motor with the given motor model.
    #[inline]
    pub const fn new(motor_model: MotorModel) -> Self {
        Self {
            enabled: true,
            target_velocity: 0.0,
            target_position: 0.0,
            max_force: f32::MAX,
            motor_model,
        }
    }

    /// Creates a new disabled linear motor with the given motor model.
    ///
    /// To enable the motor later, use [`set_enabled`](Self::set_enabled).
    #[inline]
    pub const fn new_disabled(motor_model: MotorModel) -> Self {
        Self {
            enabled: false,
            ..Self::new(motor_model)
        }
    }

    /// Enables or disables the motor.
    #[inline]
    pub const fn set_enabled(&mut self, enabled: bool) -> &mut Self {
        self.enabled = enabled;
        self
    }

    /// Sets the target linear velocity in meters per second.
    #[inline]
    pub const fn with_target_velocity(mut self, velocity: f32) -> Self {
        self.target_velocity = velocity;
        self
    }

    /// Sets the target position.
    #[inline]
    pub const fn with_target_position(mut self, target_position: f32) -> Self {
        self.target_position = target_position;
        self
    }

    /// Sets the maximum force the motor can apply.
    #[inline]
    pub const fn with_max_force(mut self, max_force: f32) -> Self {
        self.max_force = max_force;
        self
    }

    /// Sets the motor model used for computing the motor force.
    #[inline]
    pub const fn with_motor_model(mut self, motor_model: MotorModel) -> Self {
        self.motor_model = motor_model;
        self
    }
}

/// A motor for driving the relative rotation of a [`SphericalJoint`].
///
/// The motor drives the rotation of the second body's [`JointFrame`] relative to the first
/// body's frame toward [`target_rotation`](Self::target_rotation), and their relative angular
/// velocity toward [`target_velocity`](Self::target_velocity), with a torque no greater than
/// [`max_torque`](Self::max_torque) about the axis it turns them. This is the three-axis
/// counterpart of [`AngularMotor`], useful for driving ball joints such as the hips and
/// shoulders of an animated character with limited strength.
///
/// [`SphericalJoint`]: crate::dynamics::joints::SphericalJoint
/// [`JointFrame`]: crate::dynamics::joints::JointFrame
#[cfg(feature = "3d")]
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Debug, PartialEq)]
pub struct SphericalMotor {
    /// Whether the motor is enabled.
    pub enabled: bool,
    /// The target rotation of the second frame relative to the first.
    pub target_rotation: Quat,
    /// The target relative angular velocity, in the first frame's coordinates (rad/s).
    pub target_velocity: Vec3,
    /// The maximum torque the motor can apply (N·m).
    pub max_torque: f32,
    /// The motor model used for computing the motor torque.
    pub motor_model: MotorModel,
    /// Whether the motor drives the swing alone, leaving the twist about the joint's
    /// [`twist_axis`](crate::dynamics::joints::SphericalJoint::twist_axis) free: it turns the
    /// second frame's twist axis onto the target's, by the shortest arc, and no more.
    pub free_twist: bool,
}

#[cfg(feature = "3d")]
impl Default for SphericalMotor {
    fn default() -> Self {
        Self::new_disabled(MotorModel::DEFAULT)
    }
}

#[cfg(feature = "3d")]
impl SphericalMotor {
    /// Creates a new enabled motor with the given model, holding the frames aligned.
    #[inline]
    pub const fn new(motor_model: MotorModel) -> Self {
        Self {
            enabled: true,
            target_rotation: Quat::IDENTITY,
            target_velocity: Vec3::ZERO,
            max_torque: f32::MAX,
            motor_model,
            free_twist: false,
        }
    }

    /// Creates a new disabled motor with the given model.
    #[inline]
    pub const fn new_disabled(motor_model: MotorModel) -> Self {
        Self {
            enabled: false,
            ..Self::new(motor_model)
        }
    }

    /// Drives the swing alone, leaving the twist free (see [`free_twist`](Self::free_twist)).
    #[inline]
    pub const fn with_free_twist(mut self) -> Self {
        self.free_twist = true;
        self
    }

    /// Sets the target rotation of the second frame relative to the first.
    #[inline]
    pub const fn with_target_rotation(mut self, target_rotation: Quat) -> Self {
        self.target_rotation = target_rotation;
        self
    }

    /// Sets the target relative angular velocity, in the first frame's coordinates.
    #[inline]
    pub const fn with_target_velocity(mut self, target_velocity: Vec3) -> Self {
        self.target_velocity = target_velocity;
        self
    }

    /// Sets the maximum torque the motor can apply.
    #[inline]
    pub const fn with_max_torque(mut self, max_torque: f32) -> Self {
        self.max_torque = max_torque;
        self
    }
}
