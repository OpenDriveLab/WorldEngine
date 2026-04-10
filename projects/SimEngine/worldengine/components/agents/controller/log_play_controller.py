from worldengine.components.agents.controller.abstract_controller import AbstractController


class LogPlayController(AbstractController):
    """
    Assume tracking controller is absolutely perfect, and just follow a trajectory.
    """

    def step(self):
        """Inherited, see superclass."""

        waypoints = self.agent.trajectory.waypoints
        velocities = self.agent.trajectory.velocities
        headings = self.agent.trajectory.headings
        angular_velocities = self.agent.trajectory.angular_velocities

        if len(waypoints) == 1:
            self.agent.set_position(waypoints[0])
            self.agent.set_velocity(velocities[0], in_local_frame=False)
            self.agent.set_heading_theta(headings[0])
            self.agent.set_angular_velocity(angular_velocities[0])
        else:
            self.agent.set_position(waypoints[1])
            self.agent.set_velocity(velocities[1], in_local_frame=False)
            self.agent.set_heading_theta(headings[1])
            self.agent.set_angular_velocity(angular_velocities[1])
