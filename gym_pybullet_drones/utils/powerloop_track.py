import math
import numpy as np
import pybullet as p


class PowerloopTrack:
    """Powerloop drone racing track with visual entry indicators and configurable episode length.
    Includes collision-free virtual waypoints for dense RL rewards.
    Gate opening: 1.0 x 1.0 m
    """

    GATE_SIZE = 1.0
    FRAME_THICKNESS = 0.08
    FRAME_DEPTH = 0.12

    # Colores visuales (RGBA)
    COLOR_ENTRY_ACTIVE = [0.0, 1.0, 0.2, 1.0]  # Verde brillante (Objetivo)
    COLOR_ENTRY_INACTIVE = [0.1, 0.4, 0.1, 0.5]  # Verde tenue/transparente
    COLOR_FRAME_BASE = [0.2, 0.25, 0.3, 1.0]  # Gris oscuro (Estructura)

    # Lista unificada de metas. Formato:
    # (x, y, z, yaw, name, type, physical_id)
    TRACK_POINTS = [
        ( 2.000,  3.500, 0.75 , -math.pi / 2, "Gate 0", "gate", 0),
        #( 3.250,  5.000, 0.75, -math.pi/ 2 , "WP 0-1", "waypoint", None),
        (-1.500,  3.500, 2.00 , math.pi / 2, "Gate 1", "gate", 1),
        #(-3.500,  3.500, 1.200, 3 * math.pi / 8, "WP 1-2", "waypoint", None),
        (-0.625,  2.000, 0.75 , -math.pi / 2, "Gate 2", "gate", 2),
        #( 0.000,  1.000, 0.75 , math.pi / 2, "WP 2-3", "waypoint", None),
        ( 0.625,  0.000, 0.75 , -math.pi / 2, "Gate 3", "gate", 3),
        #( 1.237, -1.750, 1.375, 5 * math.pi / 8, "WP 3-4", "waypoint", None),
        (-1.500, -3.500, 2.00 , 3 * math.pi / 4, "Gate 4", "gate", 4),
        #(-1.250, -4.500, 1.375, -3 * math.pi / 4, "WP 4-5", "waypoint", None),
        ( 2.000, -3.500, 0.75 , -math.pi / 2, "Gate 5", "gate", 5),
        #( 3.312, -1.750, 0.75 , -math.pi / 2, "WP 5-6", "waypoint", None),
        ( 0.625,  0.000, 0.75 , math.pi / 2, "Gate 6", "gate", 3), # Usa la puerta 3 física
        #( 0.312,  1.750, 0.75 , math.pi / 2, "WP 6-0", "waypoint", None),
    ]

    def __init__(
        self,
        show_labels=True,
        enable_collisions=True,
        draw_debug_lines=True,
        max_gates: int | None = None,
        physics_client_id: int = 0,
        scale: float = 1.0,
        position_variation: float = 0.02,
        orientation_variation: float = 0.02,
        seed: int = 0,
    ):
        """Create a seeded variant; scale affects XY, not gate dimensions or height.

        Position variation is a fraction of the base track dimensions, while
        orientation variation is a fraction of pi. The variant stays fixed
        for the lifetime of this track.
        """
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be a finite positive number")
        if (
            not math.isfinite(position_variation)
            or not math.isfinite(orientation_variation)
            or position_variation < 0.0
            or orientation_variation < 0.0
        ):
            raise ValueError("track variations must be non-negative")

        self.show_labels = show_labels
        self.enable_collisions = enable_collisions
        self.draw_debug_lines = draw_debug_lines
        self.max_gates = max_gates
        self.physics_client_id = physics_client_id
        self.scale = scale
        self.position_variation = position_variation
        self.orientation_variation = orientation_variation
        self.seed = seed
        self.track_points = self._make_track_points(seed)

        self.gate_structures = []
        self.debug_item_ids = []

        self.current_step = 0
        self.current_gate_idx = 0

    def _make_track_points(self, seed):
        """Scale the route and apply reproducible pose jitter without scaling gates."""
        rng = np.random.RandomState(seed)
        positions = np.asarray(
            [(point[0], point[1], point[2]) for point in self.TRACK_POINTS],
            dtype=np.float64,
        )
        variation_extent = np.ptp(positions, axis=0) * self.position_variation
        yaw_extent = math.pi * self.orientation_variation

        physical_gate_offsets = {}
        physical_yaw_offsets = {}
        varied_points = []

        for x, y, z, yaw, name, point_type, physical_id in self.TRACK_POINTS:
            position = np.array([x, y, z], dtype=np.float64)
            position[:2] *= self.scale

            if point_type == "gate":
                if physical_id not in physical_gate_offsets:
                    physical_gate_offsets[physical_id] = rng.uniform(
                        -variation_extent,
                        variation_extent,
                    )
                    physical_yaw_offsets[physical_id] = rng.uniform(
                        -yaw_extent,
                        yaw_extent,
                    )
                position += physical_gate_offsets[physical_id] * np.array(
                    [self.scale, self.scale, 1.0]
                )
                yaw += physical_yaw_offsets[physical_id]
            else:
                position += rng.uniform(
                    -variation_extent,
                    variation_extent,
                ) * np.array([self.scale, self.scale, 1.0])
                yaw += rng.uniform(-yaw_extent, yaw_extent)

            varied_points.append(
                (
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    float(yaw),
                    name,
                    point_type,
                    physical_id,
                )
            )

        return tuple(varied_points)

    def create_bar(self, position, size, yaw, color=None):
        """Crea una barra primitiva en PyBullet y retorna su body ID."""
        if color is None:
            color = self.COLOR_FRAME_BASE

        half_extents = [size[0] / 2, size[1] / 2, size[2] / 2]

        collision = (
            p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=self.physics_client_id,
            )
            if self.enable_collisions
            else -1
        )

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self.physics_client_id,
        )
        orientation = p.getQuaternionFromEuler([0, 0, yaw])

        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=self.physics_client_id,
        )

    def create_virtual_waypoint(self, x, y, z, name, target_idx):
        """Crea un indicador visual esférico sin colisiones físicas."""
        initial_color = (
            self.COLOR_ENTRY_ACTIVE
            if target_idx == self.current_gate_idx
            else self.COLOR_ENTRY_INACTIVE
        )
        
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.15,
            rgbaColor=initial_color,
            physicsClientId=self.physics_client_id,
        )
        
        # baseCollisionShapeIndex = -1 asegura que el dron lo atraviese libremente
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, z],
            physicsClientId=self.physics_client_id,
        )

        if self.show_labels:
            txt_id = p.addUserDebugText(
                name,
                [x, y, z + 0.4],
                textColorRGB=[1, 1, 1],
                textSize=1.0,
                physicsClientId=self.physics_client_id,
            )
            self.debug_item_ids.append(txt_id)

        self.gate_structures.append({
            "base_bodies": [],
            "front_bodies": [body_id],
            "index": target_idx,
        })

    def create_gate(self, x, y, z, yaw, name, gate_idx):
        """Construye un gate de 4 barras, registrando partes activas y estáticas."""
        half = self.GATE_SIZE / 2
        front_offset = -self.FRAME_DEPTH / 2 + 0.01
        front_thickness = 0.02

        horizontal_front = (self.GATE_SIZE + 2 * self.FRAME_THICKNESS, front_thickness, self.FRAME_THICKNESS)
        vertical_front = (self.FRAME_THICKNESS, front_thickness, self.GATE_SIZE)
        horizontal_size = (self.GATE_SIZE + 2 * self.FRAME_THICKNESS, self.FRAME_DEPTH, self.FRAME_THICKNESS)
        vertical_size = (self.FRAME_THICKNESS, self.FRAME_DEPTH, self.GATE_SIZE)

        base_bars = [
            ((0, 0, half + self.FRAME_THICKNESS / 2), horizontal_size, self.COLOR_FRAME_BASE),
            ((0, 0, -half - self.FRAME_THICKNESS / 2), horizontal_size, self.COLOR_FRAME_BASE),
            ((-half - self.FRAME_THICKNESS / 2, 0, 0), vertical_size, self.COLOR_FRAME_BASE),
            ((half + self.FRAME_THICKNESS / 2, 0, 0), vertical_size, self.COLOR_FRAME_BASE),
        ]

        front_bars = [
            ((0, front_offset, half + self.FRAME_THICKNESS / 2), horizontal_front),
            ((0, front_offset, -half - self.FRAME_THICKNESS / 2), horizontal_front),
            ((-half - self.FRAME_THICKNESS / 2, front_offset, 0), vertical_front),
            ((half + self.FRAME_THICKNESS / 2, front_offset, 0), vertical_front),
        ]

        c, s = math.cos(yaw), math.sin(yaw)
        base_bodies = []
        for local_position, size, color in base_bars:
            lx, ly, lz = local_position
            wx = x + c * lx - s * ly
            wy = y + s * lx + c * ly
            wz = z + lz
            body_id = self.create_bar([wx, wy, wz], size, yaw, color=color)
            base_bodies.append(body_id)

        front_bodies = []
        is_active = gate_idx == self.current_gate_idx
        initial_color = self.COLOR_ENTRY_ACTIVE if is_active else self.COLOR_ENTRY_INACTIVE

        for local_position, size in front_bars:
            lx, ly, lz = local_position
            wx = x + c * lx - s * ly
            wy = y + s * lx + c * ly
            wz = z + lz
            body_id = self.create_bar([wx, wy, wz], size, yaw, color=initial_color)
            front_bodies.append(body_id)

        if self.show_labels:
            txt_id = p.addUserDebugText(
                name,
                [x, y, z + 0.8],
                textColorRGB=[1, 1, 1],
                textSize=1.2,
                physicsClientId=self.physics_client_id,
            )
            self.debug_item_ids.append(txt_id)

        if self.draw_debug_lines:
            normal = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float32)
            self.add_entry_visual_indicators([x, y, z], normal)

        self.gate_structures.append({
            "base_bodies": base_bodies,
            "front_bodies": front_bodies,
            "index": gate_idx,
        })

    def add_entry_visual_indicators(self, gate_position, gate_normal, color_entry=[0.0, 1.0, 0.2]):
        """Dibuja flechas indicadoras en 3D."""
        pos = np.array(gate_position, dtype=np.float32)
        normal = np.array(gate_normal, dtype=np.float32)
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        arrow_start = pos - normal * 0.8
        arrow_end = pos + normal * 0.2

        line_id = p.addUserDebugLine(
            arrow_start,
            arrow_end,
            color_entry,
            lineWidth=4,
            physicsClientId=self.physics_client_id,
        )
        txt_id = p.addUserDebugText(
            "IN ->",
            arrow_start + np.array([0, 0, 0.2]),
            textColorRGB=color_entry,
            textSize=1.1,
            physicsClientId=self.physics_client_id,
        )

        self.debug_item_ids.extend([line_id, txt_id])

    def create(self):
        """Genera la pista completa y los waypoints virtuales."""
        for idx, (x, y, z, yaw, name, p_type, _) in enumerate(self.track_points):
            if p_type == "gate":
                self.create_gate(x, y, z, yaw, name, idx)
            else:
                self.create_virtual_waypoint(x, y, z, name, idx)
                
        self.update_gate_visuals()
        return [b for struct in self.gate_structures for b in struct["base_bodies"]]

    def update_gate_visuals(self):
        """Actualiza el resaltado visual del objetivo activo frente a los inactivos."""
        for struct in self.gate_structures:
            is_active = struct["index"] == self.current_gate_idx
            target_color = self.COLOR_ENTRY_ACTIVE if is_active else self.COLOR_ENTRY_INACTIVE
            for body_id in struct["front_bodies"]:
                p.changeVisualShape(
                    body_id,
                    -1,
                    rgbaColor=target_color,
                    physicsClientId=self.physics_client_id,
                )

    def advance_gate(self) -> np.ndarray:
        """Avanza al siguiente objetivo del circuito."""
        self.current_step += 1
        self.current_gate_idx = self.current_step % len(self.TRACK_POINTS)
        self.update_gate_visuals()
        return self.get_current_target()

    def is_completed(self) -> bool:
        if self.max_gates is None:
            return False
        return self.current_step >= self.max_gates

    def get_progress(self) -> float:
        if self.max_gates is None:
            return 0.0
        return min(1.0, self.current_step / float(self.max_gates))

    def reset(self):
        active_bodies = {
            p.getBodyUniqueId(i, physicsClientId=self.physics_client_id)
            for i in range(p.getNumBodies(physicsClientId=self.physics_client_id))
        }
        for struct in self.gate_structures:
            for body_id in struct["base_bodies"] + struct["front_bodies"]:
                if body_id in active_bodies:
                    p.removeBody(body_id, physicsClientId=self.physics_client_id)
        for debug_id in self.debug_item_ids:
            p.removeUserDebugItem(
                debug_id, physicsClientId=self.physics_client_id
            )

        self.gate_structures.clear()
        self.debug_item_ids.clear()
        self.current_step = 0
        self.current_gate_idx = 0

    @classmethod
    def get_gate_positions(cls, track_points=None):
        points = cls.TRACK_POINTS if track_points is None else track_points
        return [pt[:5] for pt in points]

    @classmethod
    def get_gate_data(cls, track_points=None):
        points = cls.TRACK_POINTS if track_points is None else track_points
        gate_data = []
        for idx, (x, y, z, yaw, name, p_type, phys_id) in enumerate(points):
            normal = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float32)
            normal /= (np.linalg.norm(normal) + 1e-8)
            gate_data.append({
                "position": np.array([x, y, z], dtype=np.float32),
                "normal": normal,
                "yaw": yaw,
                "name": name,
                "type": p_type,
                # SOLUCIÓN: Si phys_id es None (waypoint), devolvemos -1
                "physical_id": phys_id if phys_id is not None else -1 
            })
        return gate_data

    @classmethod
    def get_waypoints(cls, track_points=None) -> list[np.ndarray]:
        points = cls.TRACK_POINTS if track_points is None else track_points
        return [
            np.array([x, y, z], dtype=np.float32)
            for x, y, z, _, _, _, _ in points
        ]

    def get_current_target(self) -> np.ndarray:
        waypoints = self.get_waypoints(self.track_points)
        return waypoints[self.current_gate_idx]