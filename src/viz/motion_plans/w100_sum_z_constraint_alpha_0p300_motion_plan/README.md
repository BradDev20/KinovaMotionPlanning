# Motion Plan Package: alpha_0p300

Generated from Pareto search trajectory optimization results.

## Source Information

- **Experiment**: w100_sum_z_constraint
- **Trajectory ID**: alpha_0p300
- **Alpha (length weight)**: 0.300
- **Length cost**: 2.1580
- **Obstacle cost**: 4.8892
- **Generated**: 2025-08-27 08:07:51

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

**Balanced Trajectory**: Balances safety and efficiency.
This trajectory offers a compromise between obstacle avoidance and path length.
