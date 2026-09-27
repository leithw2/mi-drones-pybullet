import pybullet as p
import math


class PowerloopTrack:
    """
    Powerloop drone racing track.

    Benchmark track based on the Powerloop configuration.
    Gates are constructed using primitive PyBullet boxes.

    Gate opening: 1.0 x 1.0 m
    """

    GATE_SIZE = 1.0
    FRAME_THICKNESS = 0.08
    FRAME_DEPTH = 0.12

    GATES = [
        # x,       y,       z,       yaw,        name
        ( 2.000,   3.500,   0.75,   -math.pi / 2, "Gate 0"),
        (-1.500,   3.500,   2.00,    math.pi / 4, "Gate 1"),
        (-0.625,   0.000,   0.75,    math.pi / 2, "Gate 2"),
        ( 0.625,   0.000,   0.75,    math.pi / 2, "Gate 3"),
        (-1.500,  -3.500,   2.00,   3 * math.pi / 4, "Gate 4"),
        ( 2.000,  -3.500,   0.75,   -math.pi / 2, "Gate 5"),
        ( 0.625,   0.000,   0.75,   -math.pi / 2, "Gate 6"),
    ]

    def __init__(self, show_labels=True):
        self.show_labels = show_labels
        self.gate_bodies = []

    def create_bar(self, position, size, yaw):
        """
        Create one primitive box used as part of a gate.
        """

        half_extents = [
            size[0] / 2,
            size[1] / 2,
            size[2] / 2
        ]

        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.1, 0.8, 0.1, 1.0]
        )

        orientation = p.getQuaternionFromEuler(
            [0, 0, yaw]
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            baseOrientation=orientation
        )

        self.gate_bodies.append(body_id)

        return body_id

    def create_gate(self, x, y, z, yaw, name):
        """
        Create a 1 x 1 m square gate.
        """

        half = self.GATE_SIZE / 2

        horizontal_size = (
            self.GATE_SIZE + 2 * self.FRAME_THICKNESS,
            self.FRAME_DEPTH,
            self.FRAME_THICKNESS
        )

        vertical_size = (
            self.FRAME_THICKNESS,
            self.FRAME_DEPTH,
            self.GATE_SIZE
        )

        bars = [
            (
                (0, 0, half + self.FRAME_THICKNESS / 2),
                horizontal_size
            ),
            (
                (0, 0, -half - self.FRAME_THICKNESS / 2),
                horizontal_size
            ),
            (
                (-half - self.FRAME_THICKNESS / 2, 0, 0),
                vertical_size
            ),
            (
                (half + self.FRAME_THICKNESS / 2, 0, 0),
                vertical_size
            )
        ]

        c = math.cos(yaw)
        s = math.sin(yaw)

        for local_position, size in bars:

            lx, ly, lz = local_position

            # Local -> world transformation
            wx = x + c * lx - s * ly
            wy = y + s * lx + c * ly
            wz = z + lz

            self.create_bar(
                [wx, wy, wz],
                size,
                yaw
            )

        if self.show_labels:

            p.addUserDebugText(
                name,
                [x, y, z + 0.8],
                textColorRGB=[1, 1, 1],
                textSize=1.2
            )

    def create(self):
        """
        Create the complete Powerloop track.

        Returns:
            list[int]: PyBullet body IDs of all gate elements.
        """

        for x, y, z, yaw, name in self.GATES:

            self.create_gate(
                x,
                y,
                z,
                yaw,
                name
            )

        return self.gate_bodies

    def reset(self):
        """
        Remove the track from the PyBullet simulation.
        """

        for body_id in self.gate_bodies:

            try:
                p.removeBody(body_id)
            except Exception:
                pass

        self.gate_bodies.clear()

    @classmethod
    def get_gate_positions(cls):
        """
        Return the gate configuration.

        Returns:
            list of tuples:
                (x, y, z, yaw, name)
        """

        return cls.GATES.copy()
