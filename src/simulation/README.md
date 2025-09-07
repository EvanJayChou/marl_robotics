## Simulation Scripts

### 1. Isaac Sim Interface (`src/simulation/isaac_sim_interface.py`)

Primary interface for physics simulation using NVIDIA's Isaac Sim.

#### Key Features:
- Robot spawning and management
- Physics simulation control
- Environment interaction
- Observation and reward computation

#### Usage Example:
```python
sim = IsaacSimInterface(config)
sim.spawn_robot("robot1", position=[0, 0, 1], orientation=[0, 0, 0])
sim.generate_terrain("rough", parameters={"amplitude": 0.2})

# Simulation loop
observations, rewards, dones, info = sim.step(actions)
```

### 2. Physics Utilities (`src/simulation/physics_utils.py`)

Utilities for physics calculations and transformations.

#### Key Features:
- Quaternion/Euler angle conversions
- Center of mass calculations
- Angular momentum computation
- Zero Moment Point (ZMP) calculations
- Support polygon computation
- Static stability checking

#### Usage Example:
```python
physics = PhysicsUtils()
zmp = physics.compute_zmp(
    com_position=com_pos,
    com_velocity=com_vel,
    com_acceleration=com_acc
)
is_stable = physics.check_static_stability(zmp, support_polygon)
```

### 3. Sensor Simulation (`src/simulation/sensor_simulation.py`)

Realistic sensor simulation with configurable noise models.

#### Supported Sensors:
- IMU (accelerometer/gyroscope)
- Force/torque sensors
- Joint encoders
- Contact sensors
- Depth cameras

#### Usage Example:
```python
sensors = SensorSimulation(config)
imu_data = sensors.simulate_imu(
    linear_acceleration=acc,
    angular_velocity=gyro,
    orientation=quat
)
```

### 4. World Generation (`src/simulation/world_generation.py`)

Procedural generation of training environments.

#### Terrain Types:
- Flat terrain
- Rough terrain (using Perlin noise)
- Stairs
- Gaps
- Slopes

#### Features:
- Configurable terrain parameters
- Obstacle placement
- Environment randomization

#### Usage Example:
```python
world_gen = WorldGeneration(config)
terrain_path = world_gen.generate_terrain(
    "rough",
    size=(10, 10),
    parameters={"amplitude": 0.2, "frequency": 0.1}
)
```
