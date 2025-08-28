# Motion Plan Package: alpha_0p100

Generated from Pareto search trajectory optimization results.

## Source Information

- **Experiment**: w100_sum_z_constraint
- **Trajectory ID**: alpha_0p100
- **Alpha (length weight)**: 0.100
- **Length cost**: 2.1724
- **Obstacle cost**: 4.8852
- **Generated**: 2025-08-27 08:08:43

## Files

- **`motion_plan.csv`**: Robot motion plan with timestamped joint angles
- **`trajectory_visualization.gif`**: Animated visualization of trajectory execution
- **`README.md`**: This summary file

## CSV Format

The motion plan CSV contains the following columns:
- `time_s`: Time in seconds
- `joint1_rad` through `joint7_rad`: Joint angles in radians

## Robot Deployment

This motion plan is ready for deployment on a Kinova Gen3 robot.
The trajectory has been validated against standard kinematic limits:
- Maximum joint velocity: 1.0 rad/s
- Maximum joint acceleration: 2.0 rad/s²

## Trajectory Characteristics

**Obstacle-Avoiding Trajectory**: Prioritizes safety over speed.
This trajectory carefully navigates around obstacles with conservative motions.
