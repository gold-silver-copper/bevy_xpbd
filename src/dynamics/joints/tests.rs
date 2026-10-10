use core::time::Duration;

use approx::assert_relative_eq;
use bevy::{mesh::MeshPlugin, prelude::*, time::TimeUpdateStrategy};

use crate::{dynamics::joints::joint_graph::JointGraph, prelude::*};

const TIMESTEP: f32 = 1.0 / 64.0;

fn create_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        PhysicsPlugins::default(),
        TransformPlugin,
        #[cfg(feature = "bevy_scene")]
        AssetPlugin::default(),
        #[cfg(feature = "bevy_scene")]
        bevy::scene::ScenePlugin,
        MeshPlugin,
    ));

    app.insert_resource(SubstepCount(20));

    app.insert_resource(Gravity(Vector::ZERO));

    app.insert_resource(Time::<Fixed>::from_duration(Duration::from_secs_f32(
        TIMESTEP,
    )));
    app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        TIMESTEP,
    )));

    app
}

/// Tests that an angular motor on a revolute joint spins the attached body.
#[test]
fn revolute_motor_spins_body() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_velocity: 2.0,
            max_torque: 100.0,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 0.0,
                damping: 10.0,
            },
            ..default()
        }),
    );

    // Initialize the app.
    app.update();

    // Run simulation for 1 second.
    let duration = 1.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let angular_velocity = body_ref.get::<AngularVelocity>().unwrap();

    #[cfg(feature = "2d")]
    {
        assert!(
            angular_velocity.0.abs() > 1.0,
            "Angular velocity should be significant"
        );
        assert_relative_eq!(angular_velocity.0, 2.0, epsilon = 0.5);
    }
    #[cfg(feature = "3d")]
    {
        let speed = angular_velocity.0.length();
        assert!(speed > 1.0, "Angular velocity should be significant");
    }
}

/// Tests that a linear motor on a prismatic joint moves the attached body.
#[test]
fn prismatic_motor_moves_body() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    app.world_mut().spawn(
        PrismaticJoint::new(anchor, dynamic)
            .with_local_anchor1(Vector::X * 2.0)
            .with_motor(LinearMotor {
                target_velocity: 1.0,
                max_force: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 0.0,
                    damping: 10.0,
                },
                ..default()
            }),
    );

    // Initialize the app.
    app.update();

    let initial_x = app.world().entity(dynamic).get::<Position>().unwrap().0.x;

    // Run simulation for 1 second.
    let duration = 1.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let final_x = body_ref.get::<Position>().unwrap().0.x;

    let displacement = final_x - initial_x;
    assert!(
        displacement > 0.5,
        "Body should have moved: {}",
        displacement
    );
}

/// Tests that an angular motor with max torque limit respects the limit.
#[test]
fn revolute_motor_respects_max_torque() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(100.0), // Heavy body to test torque limiting
            #[cfg(feature = "2d")]
            AngularInertia(100.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(100.0)),
        ))
        .id();

    // Create a revolute joint with a very limited motor torque.
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_velocity: 10.0,
            max_torque: 0.1,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 0.0,
                damping: 1.0,
            },
            ..default()
        }),
    );

    // Initialize the app.
    app.update();

    // Run simulation for 1 second.
    let duration = 1.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let angular_velocity = body_ref.get::<AngularVelocity>().unwrap();

    #[cfg(feature = "2d")]
    {
        assert!(
            angular_velocity.0.abs() < 5.0,
            "Velocity should be limited by max torque"
        );
    }
    #[cfg(feature = "3d")]
    {
        let speed = angular_velocity.0.length();
        assert!(speed < 5.0, "Velocity should be limited by max torque");
    }
}

/// Tests that a position-targeting motor moves the joint towards the target position.
#[test]
fn revolute_motor_position_target() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Create a revolute joint with a position-targeting motor.
    let target_angle = 1.0;
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_position: target_angle,
            max_torque: f32::MAX,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 50.0,
                damping: 20.0,
            },
            ..default()
        }),
    );

    // Initialize the app.
    app.update();

    // Run simulation for 3 seconds to let it settle.
    let duration = 3.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let rotation = body_ref.get::<Rotation>().unwrap();

    // The body should have rotated towards the target angle (allow some tolerance).
    #[cfg(feature = "2d")]
    {
        let angle = rotation.as_radians();
        assert!(
            angle.abs() > 0.3,
            "Motor should have rotated the body: {}",
            angle
        );
    }
    #[cfg(feature = "3d")]
    {
        let (axis, angle) = rotation.to_axis_angle();
        let signed_angle = angle * axis.z.signum();
        assert!(
            signed_angle.abs() > 0.3,
            "Motor should have rotated the body: {}",
            signed_angle
        );
    }
}

/// Tests that a linear position-targeting motor moves the joint towards the target position.
#[test]
fn prismatic_motor_position_target() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Create a prismatic joint with a position-targeting motor.
    // Use AccelerationBased model for stable position targeting.
    let target_position = 1.0; // Target is 1 meter along the slider axis
    app.world_mut().spawn(
        PrismaticJoint::new(anchor, dynamic)
            .with_local_anchor1(Vector::X * 2.0)
            .with_motor(LinearMotor {
                target_position,
                max_force: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 10.0,
                    damping: 5.0,
                },
                ..default()
            }),
    );

    // Initialize the app.
    app.update();

    let initial_pos = app.world().entity(dynamic).get::<Position>().unwrap().0;

    // Run simulation for 3 seconds to let it settle.
    let duration = 3.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let final_pos = body_ref.get::<Position>().unwrap().0;

    assert!(!final_pos.x.is_nan(), "Final position should not be NaN");
    assert!(!final_pos.y.is_nan(), "Final position should not be NaN");

    let displacement = final_pos.x - initial_pos.x;

    assert!(
        displacement.abs() > 0.1 || final_pos.x.abs() > 0.1,
        "Body should have moved: displacement={}, final_x={}",
        displacement,
        final_pos.x
    );
}

/// Tests that a velocity motor on a revolute joint respects angle limits.
///
/// The motor drives with constant velocity, but the joint should stop
/// when it reaches the angle limit.
#[test]
fn revolute_motor_respects_angle_limits() {
    use core::f32::consts::PI;

    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    let angle_limit = PI / 4.0;

    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic)
            .with_angle_limits(-angle_limit, angle_limit)
            .with_motor(AngularMotor {
                target_velocity: 5.0,
                max_torque: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 0.0,
                    damping: 10.0,
                },
                ..default()
            }),
    );

    app.update();

    // Run for 2 seconds - enough time for motor to hit the limit.
    let duration = 2.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let rotation = body_ref.get::<Rotation>().unwrap();

    #[cfg(feature = "2d")]
    {
        let angle = rotation.as_radians();
        assert!(
            angle <= angle_limit + 0.1,
            "Angle {} should not exceed limit {}",
            angle,
            angle_limit
        );
        assert!(
            angle > angle_limit - 0.3,
            "Angle {} should be near the limit {}",
            angle,
            angle_limit
        );
    }
    #[cfg(feature = "3d")]
    {
        let (axis, angle) = rotation.to_axis_angle();
        let signed_angle = angle * axis.z.signum();
        assert!(
            signed_angle <= angle_limit + 0.1,
            "Angle {} should not exceed limit {}",
            signed_angle,
            angle_limit
        );
        assert!(
            signed_angle > angle_limit - 0.3,
            "Angle {} should be near the limit {}",
            signed_angle,
            angle_limit
        );
    }
}

/// Tests that a velocity motor on a prismatic joint respects distance limits.
///
/// The motor drives with constant velocity, but the joint should stop
/// when it reaches the distance limit.
#[test]
fn prismatic_motor_respects_limits() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    // Start at origin so we can measure displacement clearly.
    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::ZERO),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Limit translation to [0, 1] meters along the slide axis.
    let distance_limit = 1.0;

    app.world_mut().spawn(
        PrismaticJoint::new(anchor, dynamic)
            .with_limits(0.0, distance_limit)
            .with_motor(LinearMotor {
                target_velocity: 5.0, // High velocity to ensure we hit the limit
                max_force: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 0.0,
                    damping: 10.0,
                },
                ..default()
            }),
    );

    app.update();

    // Make sure the motor is not near the limit from the start.
    {
        let body_ref = app.world().entity(dynamic);
        let position = body_ref.get::<Position>().unwrap();
        assert!(
            (position.0.x - distance_limit.real()).abs() > 0.1,
            "Displacement {} should not be near the limit {} at the start of the test",
            position.0.x,
            distance_limit
        );
    }

    // Run for 2 seconds - enough time for motor to hit the limit.
    let duration = 2.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let position = body_ref.get::<Position>().unwrap();

    // The displacement along the slide axis (X) should be at or near the limit.
    let displacement = position.x.f32();
    assert!(
        displacement <= distance_limit + 0.001,
        "Displacement {} should not exceed limit {}",
        displacement,
        distance_limit
    );
    assert!(
        (displacement - distance_limit).abs() < 0.1,
        "Displacement {} should be near the limit {}",
        displacement,
        distance_limit
    );
}

/// Tests that `ForceBased` motor model works for revolute joints.
///
/// This is the physically accurate motor model that takes mass into account.
#[test]
fn revolute_motor_force_based() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Use ForceBased motor model with velocity control.
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_velocity: 2.0,
            max_torque: 100.0,
            motor_model: MotorModel::ForceBased {
                stiffness: 0.0,
                damping: 10.0,
            },
            ..default()
        }),
    );

    app.update();

    let body_ref = app.world().entity(dynamic);
    let angular_velocity = body_ref.get::<AngularVelocity>().unwrap();
    #[cfg(feature = "2d")]
    let initial_speed = angular_velocity.0.abs();
    #[cfg(feature = "3d")]
    let initial_speed = angular_velocity.0.length();

    assert!(
        initial_speed.abs() < 0.001,
        "ForceBased motor should be initiall still, speed: {}",
        initial_speed
    );

    // Run simulation for 1 second.
    let duration = 1.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let angular_velocity = body_ref.get::<AngularVelocity>().unwrap();

    // The body should have gained angular velocity.
    #[cfg(feature = "2d")]
    {
        assert!(
            angular_velocity.0.abs() > 0.5,
            "ForceBased motor should spin the body: {}",
            angular_velocity.0
        );
    }
    #[cfg(feature = "3d")]
    {
        let speed = angular_velocity.0.length();
        assert!(
            speed > 0.5,
            "ForceBased motor should spin the body: {}",
            speed
        );
    }
}

/// Tests that the default `SpringDamper` motor model works.
///
/// `SpringDamper` is unconditionally stable and uses `frequency`/`damping_ratio`.
#[test]
fn revolute_motor_spring_damper() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    let target_angle = 1.0;
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            target_position: target_angle,
            max_torque: f32::MAX,
            motor_model: MotorModel::SpringDamper {
                frequency: 2.0,
                damping_ratio: 1.0,
            },
            ..default()
        }),
    );

    app.update();

    // TODO Assert the body is initially not near the target.

    // Run simulation for 3 seconds to let the spring settle.
    let duration = 3.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let rotation = body_ref.get::<Rotation>().unwrap();

    // The body should have rotated towards the target.
    #[cfg(feature = "2d")]
    {
        let angle = rotation.as_radians();
        assert!(
            angle.abs() > 0.3,
            "SpringDamper motor should rotate towards target: {}",
            angle
        );
    }
    #[cfg(feature = "3d")]
    {
        let (axis, angle) = rotation.to_axis_angle();
        let signed_angle = angle * axis.z.signum();
        assert!(
            signed_angle.abs() > 0.3,
            "SpringDamper motor should rotate towards target: {}",
            signed_angle
        );
    }
}

/// Tests that a motor with both velocity and position targeting works.
///
/// Combined spring-damper behavior: position targeting with velocity damping.
#[test]
fn revolute_motor_combined_position_velocity() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Motor with both position and velocity targeting.
    // The velocity adds a constant offset to the spring behavior.
    let target_angle = 0.5;
    app.world_mut().spawn(
        RevoluteJoint::new(anchor, dynamic).with_motor(AngularMotor {
            enabled: true,
            target_position: target_angle,
            target_velocity: 0.5,
            max_torque: 100.0,
            motor_model: MotorModel::AccelerationBased {
                stiffness: 30.0,
                damping: 15.0,
            },
        }),
    );

    app.update();

    // Run for 2 seconds.
    let duration = 2.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let rotation = body_ref.get::<Rotation>().unwrap();

    // The body should have rotated in the positive direction.
    #[cfg(feature = "2d")]
    {
        let angle = rotation.as_radians();
        assert!(
            angle > 0.2,
            "Combined motor should rotate body positively: {}",
            angle
        );
    }
    #[cfg(feature = "3d")]
    {
        let (axis, angle) = rotation.to_axis_angle();
        let signed_angle = angle * axis.z.signum();
        assert!(
            signed_angle > 0.2,
            "Combined motor should rotate body positively: {}",
            signed_angle
        );
    }
}

/// Tests that a prismatic motor with both velocity and position targeting works.
#[test]
fn prismatic_motor_combined_position_velocity() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::X * 2.0),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    // Motor with both position and velocity targeting.
    app.world_mut().spawn(
        PrismaticJoint::new(anchor, dynamic)
            .with_local_anchor1(Vector::X * 2.0)
            .with_motor(LinearMotor {
                enabled: true,
                target_position: 1.0,
                target_velocity: 0.5,
                max_force: 100.0,
                motor_model: MotorModel::AccelerationBased {
                    stiffness: 20.0,
                    damping: 10.0,
                },
            }),
    );

    app.update();

    let initial_x = app.world().entity(dynamic).get::<Position>().unwrap().0.x;

    // Run for 2 seconds.
    let duration = 2.0;
    let steps = (duration / TIMESTEP) as usize;

    for _ in 0..steps {
        app.update();
    }

    let body_ref = app.world().entity(dynamic);
    let final_x = body_ref.get::<Position>().unwrap().0.x;

    // The body should have moved.
    let displacement = final_x - initial_x;
    assert!(
        displacement.abs() > 0.1,
        "Combined motor should move the body: {}",
        displacement
    );
}

#[cfg(feature = "2d")]
fn z_spin(angular_velocity: &AngularVelocity) -> f32 {
    angular_velocity.0
}

#[cfg(feature = "3d")]
fn z_spin(angular_velocity: &AngularVelocity) -> f32 {
    angular_velocity.z
}

/// Tests that angular joint damping between bodies with unequal inertia conserves angular momentum.
#[test]
fn joint_damping_conserves_angular_momentum() {
    let mut app = create_app();
    app.finish();

    let light = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::ZERO),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
            #[cfg(feature = "2d")]
            AngularVelocity(10.0),
            #[cfg(feature = "3d")]
            AngularVelocity(Vector::Z * 10.0),
        ))
        .id();

    let heavy = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::ZERO),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(10.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(10.0)),
        ))
        .id();

    app.world_mut().spawn((
        RevoluteJoint::new(light, heavy),
        JointDamping {
            linear: 0.0,
            angular: 20.0,
        },
    ));

    app.update();

    let steps = (1.0 / TIMESTEP) as usize;
    for _ in 0..steps {
        app.update();
    }

    let light_spin = z_spin(app.world().get::<AngularVelocity>(light).unwrap());
    let heavy_spin = z_spin(app.world().get::<AngularVelocity>(heavy).unwrap());

    assert_relative_eq!(light_spin, heavy_spin, epsilon = 1e-3);
    assert_relative_eq!(1.0 * light_spin + 10.0 * heavy_spin, 10.0, epsilon = 1e-3);
}

/// Joints a moving kinematic body to a resting dynamic body and returns the kinematic body's
/// velocities after one second.
fn kinematic_velocity_after_damping<J: Component>(
    joint: fn(Entity, Entity) -> J,
    initial_velocity: (LinearVelocity, AngularVelocity),
    damping: JointDamping,
) -> (LinearVelocity, AngularVelocity) {
    let mut app = create_app();
    app.finish();

    let kinematic = app
        .world_mut()
        .spawn((
            RigidBody::Kinematic,
            Position(RVector::ZERO),
            // Kinematic bodies normally get mass from their colliders.
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
            initial_velocity,
        ))
        .id();

    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::ZERO),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();

    app.world_mut().spawn((joint(kinematic, dynamic), damping));

    app.update();

    let steps = (1.0 / TIMESTEP) as usize;
    for _ in 0..steps {
        app.update();
    }

    let body = app.world().entity(kinematic);
    (
        *body.get::<LinearVelocity>().unwrap(),
        *body.get::<AngularVelocity>().unwrap(),
    )
}

/// Tests that angular joint damping does not change the angular velocity of a kinematic body.
#[test]
fn joint_damping_does_not_change_kinematic_angular_velocity() {
    #[cfg(feature = "2d")]
    let spin = AngularVelocity(2.0);
    #[cfg(feature = "3d")]
    let spin = AngularVelocity(Vector::Z * 2.0);

    let (_, undamped) = kinematic_velocity_after_damping(
        RevoluteJoint::new,
        (LinearVelocity::ZERO, spin),
        JointDamping::default(),
    );
    assert_eq!(undamped, spin, "control: joint alone changed it");

    let (_, damped) = kinematic_velocity_after_damping(
        RevoluteJoint::new,
        (LinearVelocity::ZERO, spin),
        JointDamping {
            linear: 0.0,
            angular: 1.0,
        },
    );
    assert_eq!(damped, spin);
}

/// Tests that linear joint damping does not change the linear velocity of a kinematic body.
#[test]
fn joint_damping_does_not_change_kinematic_linear_velocity() {
    let velocity = LinearVelocity(Vector::X * 2.0);

    let (undamped, _) = kinematic_velocity_after_damping(
        PrismaticJoint::new,
        (velocity, AngularVelocity::ZERO),
        JointDamping::default(),
    );
    assert_eq!(undamped, velocity, "control: joint alone changed it");

    let (damped, _) = kinematic_velocity_after_damping(
        PrismaticJoint::new,
        (velocity, AngularVelocity::ZERO),
        JointDamping {
            linear: 1.0,
            angular: 0.0,
        },
    );
    assert_eq!(damped, velocity);
}

/// Tests that angular joint damping still damps the free axis of a body with locked rotation axes.
#[cfg(feature = "3d")]
#[test]
fn joint_damping_damps_free_axis_of_rotation_locked_body() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(RVector::ZERO)))
        .id();

    let body = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(RVector::ZERO),
            Mass(1.0),
            AngularInertia::new(Vec3::splat(1.0)),
            LockedAxes::new().lock_rotation_x().lock_rotation_z(),
            AngularVelocity(Vector::Y * 2.0),
        ))
        .id();

    app.world_mut().spawn((
        SphericalJoint::new(anchor, body),
        JointDamping {
            linear: 0.0,
            angular: 5.0,
        },
    ));

    app.update();

    let steps = (1.0 / TIMESTEP) as usize;
    for _ in 0..steps {
        app.update();
    }

    let spin = app.world().get::<AngularVelocity>(body).unwrap().y;
    assert!(spin.abs() < 0.05, "free axis was not damped: {spin}");
}

/// Tests that a thin rod tumbling on a ball joint gains neither energy nor angular momentum
/// about the joint: the joint must see the rod's inertia as the rod is turned in each
/// substep, not as it was at the start of the step.
#[cfg(feature = "3d")]
#[test]
fn tumbling_rod_on_ball_joint_conserves_momentum() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(Vector::ZERO)))
        .id();

    // A rod 1 m long along x, its end on the joint at the origin.
    let (mass, inertia) = (1.0, Vec3::new(1e-3, 0.0835, 0.0835));
    let spin = Vector::new(40.0, 20.0, 0.0);
    let centre = Vector::X * 0.5;
    let rod = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(centre),
            Mass(mass),
            AngularInertia::new(inertia),
            AngularVelocity(spin),
            LinearVelocity(spin.cross(centre)),
        ))
        .id();

    app.world_mut()
        .spawn(SphericalJoint::new(anchor, rod).with_local_anchor2(-centre));

    let state = |app: &App| {
        let body = app.world().entity(rod);
        let at = body.get::<Position>().unwrap().0;
        let turn = body.get::<Rotation>().unwrap().0;
        let v = body.get::<LinearVelocity>().unwrap().0;
        let w = body.get::<AngularVelocity>().unwrap().0;
        let world_inertia = Mat3::from_quat(turn)
            * Mat3::from_diagonal(inertia)
            * Mat3::from_quat(turn).transpose();
        let momentum = world_inertia * w + mass * at.cross(v);
        let energy = 0.5 * mass * v.length_squared() + 0.5 * w.dot(world_inertia * w);
        (momentum, energy)
    };
    let (momentum0, energy0) = state(&app);

    app.update();
    for _ in 0..(1.0 / TIMESTEP) as usize {
        app.update();
    }

    // The joint's corrections lose some energy; they must not make any. Stale inertia
    // (as turned at the step's start) gives about a megajoule here.
    let (momentum, energy) = state(&app);
    assert!(
        energy <= energy0,
        "the rod gained energy: {energy0} J -> {energy} J"
    );
    assert!(
        momentum.length() <= momentum0.length() * 1.01,
        "the rod gained angular momentum about the joint: {momentum0} -> {momentum}"
    );
}

/// Spawns two dynamic bodies and a disabled revolute joint between them, and runs a step.
fn disabled_joint_app() -> (App, [Entity; 2], Entity) {
    let mut app = create_app();
    app.finish();

    let mut body = |x: f32| {
        app.world_mut()
            .spawn((
                RigidBody::Dynamic,
                Position(RVector::X * x),
                Mass(1.0),
                #[cfg(feature = "2d")]
                AngularInertia(1.0),
                #[cfg(feature = "3d")]
                AngularInertia::new(Vec3::splat(1.0)),
            ))
            .id()
    };
    let bodies = [body(0.0), body(1.0)];
    let joint = app
        .world_mut()
        .spawn((RevoluteJoint::new(bodies[0], bodies[1]), JointDisabled))
        .id();

    app.update();
    (app, bodies, joint)
}

/// Tests that despawning a disabled joint after both of its bodies doesn't panic.
#[test]
fn disabled_joint_despawned_after_its_bodies() {
    let (mut app, bodies, joint) = disabled_joint_app();

    for body in bodies {
        app.world_mut().despawn(body);
    }
    app.world_mut().despawn(joint);
    app.update();

    assert!(app.world().resource::<JointGraph>().get(joint).is_none());
}

/// Tests that enabling a disabled joint still puts it in the joint graph.
#[test]
fn enabled_joint_joins_the_joint_graph() {
    let (mut app, _, joint) = disabled_joint_app();
    assert!(app.world().resource::<JointGraph>().get(joint).is_none());

    app.world_mut().entity_mut(joint).remove::<JointDisabled>();
    app.update();

    assert!(app.world().resource::<JointGraph>().get(joint).is_some());
}

/// Spawns a body on a spherical joint to a static anchor at its centre, driven by `motor`.
#[cfg(feature = "3d")]
fn driven_ball(app: &mut App, motor: SphericalMotor) -> Entity {
    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(Vector::ZERO)))
        .id();
    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(Vector::ZERO),
            Mass(1.0),
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();
    app.world_mut()
        .spawn(SphericalJoint::new(anchor, dynamic).with_motor(motor));
    dynamic
}

/// Tests that a spherical motor turns the body to its target rotation about any axis.
#[cfg(feature = "3d")]
#[test]
fn spherical_motor_reaches_target_rotation() {
    let mut app = create_app();
    app.finish();

    let target = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.5).normalize(), 1.2);
    let dynamic = driven_ball(
        &mut app,
        SphericalMotor::new(MotorModel::SpringDamper {
            frequency: 5.0,
            damping_ratio: 1.0,
        })
        .with_target_rotation(target),
    );

    app.update();
    for _ in 0..(2.0 / TIMESTEP) as usize {
        app.update();
    }

    let rotation = app.world().entity(dynamic).get::<Rotation>().unwrap().0;
    let off = rotation.angle_between(target);
    assert!(off < 0.01, "the body is {off} rad from the target rotation");
}

/// Tests that a spherical motor turns the body with no more than its maximum torque.
#[cfg(feature = "3d")]
#[test]
fn spherical_motor_respects_max_torque() {
    let mut app = create_app();
    app.finish();

    let max_torque = 0.5;
    let target = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 1.0).normalize(), 1.5);
    let dynamic = driven_ball(
        &mut app,
        SphericalMotor::new(MotorModel::SpringDamper {
            frequency: 20.0,
            damping_ratio: 1.0,
        })
        .with_target_rotation(target)
        .with_max_torque(max_torque),
    );

    app.update();
    let duration = 0.5;
    for _ in 0..(duration / TIMESTEP) as usize {
        app.update();
    }

    // With a unit inertia, the most the motor can turn the body is `max_torque * t^2 / 2`.
    let body = app.world().entity(dynamic);
    let turned = body.get::<Rotation>().unwrap().0.angle_between(Quat::IDENTITY);
    let most = 0.5 * max_torque * duration * duration;
    assert!(turned > 0.5 * most, "the motor turned the body only {turned} rad");
    assert!(turned < 1.1 * most, "the motor turned the body {turned} rad, past {most}");
    let speed = body.get::<AngularVelocity>().unwrap().0.length();
    assert!(speed < 1.1 * max_torque * duration, "spinning at {speed} rad/s");
}

/// Tests that a spherical joint with free points (an infinite point compliance) holds nothing of
/// the body's place, and that its motor with a free twist turns its twist axis onto the
/// target's while its spin about that axis goes on.
#[cfg(feature = "3d")]
#[test]
fn spherical_drive_with_free_points_and_twist() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(Vector::ZERO)))
        .id();
    let going = Vector::new(1.0, -2.0, 0.5);
    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(Vector::ZERO),
            Rotation(Quat::from_rotation_x(0.6)),
            LinearVelocity(going),
            AngularVelocity(Quat::from_rotation_x(0.6) * Vector::Z * 2.0),
            Mass(1.0),
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();
    let mut joint = SphericalJoint::new(anchor, dynamic)
        .with_twist_axis(Vector::Z)
        .with_motor(
            SphericalMotor::new(MotorModel::SpringDamper {
                frequency: 5.0,
                damping_ratio: 1.0,
            })
            .with_free_twist(),
        );
    joint.point_compliance = f32::INFINITY;
    app.world_mut().spawn(joint);

    app.update();
    let duration = 2.0;
    for _ in 0..(duration / TIMESTEP) as usize {
        app.update();
    }

    let body = app.world().entity(dynamic);
    let at = body.get::<Position>().unwrap().0;
    assert!(
        at.distance(going * duration) < 0.1,
        "the body went on to {at}, not {}",
        going * duration
    );
    let up = body.get::<Rotation>().unwrap().0 * Vector::Z;
    assert!(up.angle_between(Vector::Z) < 0.01, "its axis is at {up}");
    let spin = body.get::<AngularVelocity>().unwrap().0;
    assert!((spin.z - 2.0).abs() < 0.05, "its twist spin went to {spin}");
}

/// Tests that a spherical joint's twist limit holds with the joint swung past 120 degrees.
///
/// The twist is measured about the bisector of the two swing axes. If the limit let go past some
/// swing, the twist would run on unchecked there, and the whole of it would be corrected at once,
/// out of no stored energy, as the swing came back.
#[cfg(feature = "3d")]
#[test]
fn spherical_twist_limit_holds_at_a_wide_swing() {
    let mut app = create_app();
    app.finish();

    // Swung 2.4 rad (some 140 degrees) about x, twisting about its own swing axis (-z) at 2 rad/s.
    let swung = Quat::from_rotation_x(2.4);
    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(Vector::ZERO)))
        .id();
    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(Vector::ZERO),
            Rotation(swung),
            AngularVelocity(swung * Vector::NEG_Z * 2.0),
            Mass(1.0),
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();
    let mut joint = SphericalJoint::new(anchor, dynamic);
    joint.twist_limit = Some(AngleLimit::new(-0.5, 0.5));
    app.world_mut().spawn(joint);

    app.update();
    for _ in 0..(1.0 / TIMESTEP) as usize {
        app.update();
    }

    // The twist about the body's own -z, from the swing-twist decomposition of its rotation.
    let q = app.world().entity(dynamic).get::<Rotation>().unwrap().0;
    let q = if q.w < 0.0 { -q } else { q };
    let twist = 2.0 * q.xyz().dot(Vector::NEG_Z).atan2(q.w);
    assert!(
        twist.abs() < 0.55,
        "twisted {twist} rad, past the limit of 0.5"
    );
}

/// Tests that a hinge pressed far past one angle limit is held back toward that limit, not
/// driven the other way round through its range once it is more than a half turn from the
/// other limit.
#[test]
#[cfg(feature = "3d")]
fn revolute_limit_pressed_past_holds_its_side() {
    use core::f32::consts::{PI, TAU};
    let mut app = create_app();
    app.finish();

    // Limited from 0 to 2.4 rad, soft past them, flung past 2.4 at 10 rad/s (to some 3.4 rad, short of
    // a half turn from the middle of its range).
    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(Vector::ZERO)))
        .id();
    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(Vector::ZERO),
            Rotation(Quat::from_rotation_z(2.0)),
            AngularVelocity(Vector::Z * 10.0),
            Mass(1.0),
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();
    let mut joint = RevoluteJoint::new(anchor, dynamic).with_angle_limits(0.0, 2.4);
    joint.limit_compliance = 0.01;
    app.world_mut().spawn(joint);

    app.update();
    // The hinge's angle followed round continuously from where it set off.
    let (mut angle, mut most) = (2.0 as f32, 0.0 as f32);
    for _ in 0..(1.0 / TIMESTEP) as usize {
        app.update();
        let q = app.world().entity(dynamic).get::<Rotation>().unwrap().0;
        let now = 2.0 * q.z.atan2(q.w);
        angle += (now - angle + PI).rem_euclid(TAU) - PI;
        most = most.max(angle);
    }
    assert!(
        most < 4.0,
        "the hinge went on round to {most} rad, past a half turn from its range's middle"
    );
}

/// Tests that a joint's forces are the step's: a body going at 1 m/s stopped by a fixed joint to a
/// static body, what its forces add up to over the steps is the momentum it took.
#[test]
fn joint_forces_add_up_to_the_momentum_taken() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(Vector::ZERO)))
        .id();
    let going = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(Vector::ZERO),
            LinearVelocity(Vector::X),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();
    let joint = app
        .world_mut()
        .spawn((FixedJoint::new(anchor, going), JointForces::new()))
        .id();

    let mut taken = Vector::ZERO;
    for _ in 0..(1.0 / TIMESTEP) as usize {
        app.update();
        taken += app.world().entity(joint).get::<JointForces>().unwrap().force() * TIMESTEP;
    }
    // The forces are on the first body, the static one: the body's push on it.
    assert!(
        (taken.x - 1.0).abs() < 0.02,
        "its forces took {taken} N·s of the 1 N·s"
    );
}

/// Tests that a fixed joint holds with no more than its most force: a 1 kg body going at 1 m/s from
/// a static body it is joined to by a joint of 2 N goes on, slowed at 2 m/s².
#[test]
fn fixed_joint_gives_past_its_most_force() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(Vector::ZERO)))
        .id();
    let going = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(Vector::ZERO),
            LinearVelocity(Vector::X),
            Mass(1.0),
            #[cfg(feature = "2d")]
            AngularInertia(1.0),
            #[cfg(feature = "3d")]
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();
    app.world_mut()
        .spawn(FixedJoint::new(anchor, going).with_max_force(2.0));

    app.update();
    for _ in 0..(0.25 / TIMESTEP) as usize {
        app.update();
    }
    let v = app.world().entity(going).get::<LinearVelocity>().unwrap().0.x;
    assert!((v - 0.5).abs() < 0.05, "it goes at {v} m/s");
}

/// Tests that a hinge turned about in the world keeps no push about its own axis from holding its
/// axes aligned: the align impulse taken as the bodies' spin across the hinge was stopped, the
/// hinge then carried round by a spin about another axis, a weak motor still holds its angle.
#[test]
#[cfg(feature = "3d")]
fn revolute_turned_about_keeps_its_angle() {
    let mut app = create_app();
    app.finish();

    let first = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(Vector::ZERO),
            AngularVelocity(Vector::X * 3.0),
            Mass(1.0),
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();
    let second = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(Vector::ZERO),
            AngularVelocity(Vector::new(3.0, 6.0, 0.0)),
            Mass(1.0),
            AngularInertia::new(Vec3::splat(1.0)),
        ))
        .id();
    let joint = RevoluteJoint::new(first, second).with_motor(
        AngularMotor::new(MotorModel::SpringDamper {
            frequency: 2.0,
            damping_ratio: 1.0,
        })
        .with_max_torque(0.5),
    );
    app.world_mut().spawn(joint);

    app.update();
    for _ in 0..(1.0 / TIMESTEP) as usize {
        app.update();
    }

    let turn = |e: Entity| app.world().entity(e).get::<Rotation>().unwrap().0;
    let relative = turn(first).inverse() * turn(second);
    let angle = 2.0 * relative.z.atan2(relative.w);
    assert!(angle.abs() < 0.1, "the hinge went to {angle} rad");
}

/// Tests that a body swinging and twisting against a spherical joint's swing and twist limits
/// gains no energy from them.
#[test]
#[cfg(feature = "3d")]
fn spherical_limits_feed_no_energy() {
    let mut app = create_app();
    app.finish();

    let anchor = app
        .world_mut()
        .spawn((RigidBody::Static, Position(Vector::ZERO)))
        .id();
    // An arm a metre long off the joint, swung about a wide axis and twisted at once.
    let dynamic = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Position(Vector::Y),
            AngularVelocity(Vector::new(6.0, 3.0, 4.0)),
            Mass(1.0),
            AngularInertia::new(Vec3::new(0.1, 0.02, 0.1)),
        ))
        .id();
    let mut joint = SphericalJoint::new(anchor, dynamic).with_local_anchor2(Vector::NEG_Y);
    joint.swing_limit = Some(AngleLimit::new(-1.2, 1.2));
    joint.twist_limit = Some(AngleLimit::new(-0.4, 0.4));
    joint.swing_compliance = 0.005;
    joint.twist_compliance = 0.005;
    app.world_mut().spawn(joint);

    let energy = |app: &App| {
        let e = app.world().entity(dynamic);
        let (v, w) = (
            e.get::<LinearVelocity>().unwrap().0,
            e.get::<AngularVelocity>().unwrap().0,
        );
        let r = e.get::<Rotation>().unwrap().0;
        let i = r * (Vec3::new(0.1, 0.02, 0.1) * (r.inverse() * w));
        0.5 * v.length_squared() + 0.5 * w.dot(i)
    };
    app.update();
    let start = energy(&app);
    let mut most = start;
    for _ in 0..(2.0 / TIMESTEP) as usize {
        app.update();
        most = most.max(energy(&app));
    }
    assert!(
        most < start * 1.05,
        "the energy rose from {start} to {most}"
    );
}

/// Tests that limbs lying on the ground with their joints pressed against their limits come
/// to rest and fall asleep: joints solved apart from the contacts kept them creeping.
#[cfg(all(feature = "3d", feature = "default-collider"))]
#[test]
fn limbs_lying_against_their_limits_sleep() {
    let mut app = create_app();
    app.insert_resource(Gravity(Vector::NEG_Z * 9.81));
    app.finish();

    let world = app.world_mut();
    world.spawn((RigidBody::Static, Collider::half_space(Vector::Z)));
    // Three capsules end to end along x on the ground, the joints between them bent short of
    // the limits they are held to: a ball whose cone is turned off the line, a hinge about z
    // that must fold.
    let limb = |world: &mut World, x: f32| {
        world
            .spawn((
                RigidBody::Dynamic,
                Position(Vector::new(x, 0.0, 0.05)),
                Collider::capsule_endpoints(
                    0.05,
                    Vector::new(-0.2, 0.0, 0.0),
                    Vector::new(0.2, 0.0, 0.0),
                ),
                Friction::new(0.6),
            ))
            .id()
    };
    let limbs = [limb(world, 0.0), limb(world, 0.4), limb(world, 0.8)];
    let mut ball = SphericalJoint::new(limbs[0], limbs[1])
        .with_local_frame1(Isometry::new(
            Vector::X * 0.2,
            Quat::from_rotation_z(0.6),
        ))
        .with_local_frame2(Isometry::new(Vector::NEG_X * 0.2, Quat::IDENTITY))
        .with_twist_axis(Vector::Z);
    ball.swing_limit = Some(AngleLimit::new(-0.3, 0.3));
    ball.twist_limit = Some(AngleLimit::new(-0.2, 0.2));
    (ball.swing_compliance, ball.twist_compliance) = (0.005, 0.005);
    let hinge = RevoluteJoint::new(limbs[1], limbs[2])
        .with_local_anchor1(Vector::X * 0.2)
        .with_local_anchor2(Vector::NEG_X * 0.2)
        .with_angle_limits(0.4, 1.2)
        .with_limit_compliance(0.005);
    world.spawn((ball, JointCollisionDisabled));
    world.spawn((hinge, JointCollisionDisabled));

    for _ in 0..(6.0 / TIMESTEP) as usize {
        app.update();
    }
    for limb in limbs {
        let e = app.world().entity(limb);
        assert!(
            e.contains::<Sleeping>(),
            "a limb still moving at {} m/s, {} rad/s",
            e.get::<LinearVelocity>().unwrap().length(),
            e.get::<AngularVelocity>().unwrap().length()
        );
    }
}
