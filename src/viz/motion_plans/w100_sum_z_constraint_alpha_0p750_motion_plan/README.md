# Motion Plan Package: alpha_0p750

Generated from Pareto search trajectory optimization results.

## Source Information

- **Experiment**: w100_sum_z_constraint
- **Trajectory ID**: alpha_0p750
- **Alpha (length weight)**: 0.750
- **Length cost**: 1.7821
- **Obstacle cost**: 5.6050
- **Generated**: 2025-08-27 08:10:41

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

**Direct Trajectory**: Prioritizes speed and efficiency.
This trajectory takes a more direct path, accepting higher obstacle proximity.
