"""Reduced planar jumper model exported through `biobuddy` when available."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from synchro_jump.modeling.athlete import AthleteMorphology


def _rod_inertia(mass: float, length: float) -> tuple[float, float, float]:
    """Return a compact diagonal inertia tensor for a slender segment."""

    transverse = mass * length**2 / 12.0
    return (0.05 * transverse, transverse, transverse)


def _segment_block(
    name: str,
    parent: str,
    translation: tuple[float, float, float],
    *,
    translations: str | None = None,
    rotations: str | None = None,
    ranges_q: tuple[tuple[float, float], ...] | None = None,
    mass: float = 0.0,
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0),
    inertia: tuple[float, float, float] = (1e-6, 1e-6, 1e-6),
) -> str:
    """Serialize one segment block in `bioMod` format."""

    tx, ty, tz = translation
    out = [
        f"segment\t{name}\n",
        f"\tparent\t{parent}\n",
        "\tRTinMatrix\t1\n",
        "\tRT\n",
        f"\t\t1.000000\t0.000000\t0.000000\t{tx:.6f}\n",
        f"\t\t0.000000\t1.000000\t0.000000\t{ty:.6f}\n",
        f"\t\t0.000000\t0.000000\t1.000000\t{tz:.6f}\n",
        "\t\t0.000000\t0.000000\t0.000000\t1.000000\n",
    ]
    if translations is not None:
        out.append(f"\ttranslations\t{translations}\n")
    if rotations is not None:
        out.append(f"\trotations\t{rotations}\n")
    if ranges_q is not None:
        out.append("\trangesQ\n")
        out.extend(f"\t\t{lower:.6f}\t{upper:.6f}\n" for lower, upper in ranges_q)
    out.append(f"\tmass\t{mass:.6f}\n")
    out.append(
        f"\tCenterOfMass\t{center_of_mass[0]:.6f}\t{center_of_mass[1]:.6f}\t{center_of_mass[2]:.6f}\n"
    )
    out.append("\tinertia\n")
    out.append(f"\t\t{inertia[0]:.6f}\t0.000000\t0.000000\n")
    out.append(f"\t\t0.000000\t{inertia[1]:.6f}\t0.000000\n")
    out.append(f"\t\t0.000000\t0.000000\t{inertia[2]:.6f}\n")
    out.append("endsegment\n\n")
    return "".join(out)


def _marker_block(name: str, parent: str, position: tuple[float, float, float]) -> str:
    """Serialize one marker block in `bioMod` format."""

    return (
        f"marker\t{name}\n"
        f"\tparent\t{parent}\n"
        f"\tposition\t{position[0]:.6f}\t{position[1]:.6f}\t{position[2]:.6f}\n"
        "\ttechnical\t1\n"
        "\tanatomical\t0\n"
        "endmarker\n\n"
    )


def _contact_block(name: str, parent: str, position: tuple[float, float, float], axis: str = "z") -> str:
    """Serialize one rigid contact point in `bioMod` format."""

    return (
        f"contact\t{name}\n"
        f"\tparent\t{parent}\n"
        f"\tposition\t{position[0]:.6f}\t{position[1]:.6f}\t{position[2]:.6f}\n"
        f"\taxis\t{axis}\n"
        "endcontact\n\n"
    )


@dataclass(frozen=True)
class PlanarJumperModelDefinition:
    """Reduced three-segment planar jumper exported as a `bioMod`.

    The model uses a planar floating root with:
    - two root translations in the sagittal plane
    - one root rotation out of plane
    - two joint rotations for knee and hip
    """

    morphology: AthleteMorphology = AthleteMorphology()
    gravity: float = 9.81
    floating_base: bool = True
    include_platform_contact: bool = True

    @property
    def q_size(self) -> int:
        """Return the number of generalized coordinates."""

        return 5 if self.floating_base else 3

    @property
    def tau_size(self) -> int:
        """Return the number of actuated generalized torques."""

        return 2 if self.floating_base else 3

    @property
    def segment_masses(self) -> tuple[float, float, float]:
        """Return the reduced mass allocation across the three segments."""

        total = self.morphology.mass_kg
        return (0.12 * total, 0.24 * total, 0.64 * total)

    @property
    def crouched_joint_configuration_rad(self) -> tuple[float, float, float, float, float]:
        """Return the crouched posture before CoM alignment over the ankle."""

        flexion = math.radians(self.morphology.initial_joint_flexion_deg)
        if self.floating_base:
            return (0.0, 0.0, 0.0, -flexion, flexion)
        return (0.0, -flexion, flexion)

    def _unpack_q_values(
        self,
        q_values: tuple[float, ...] | list[float],
    ) -> tuple[float, float, float, float, float]:
        """Expand one generalized configuration to the floating-base convention."""

        if self.floating_base:
            if len(q_values) != 5:
                raise ValueError("The floating-base jumper expects 5 generalized coordinates")
            q_root_x, q_root_z, q_root_rot, q_knee, q_hip = q_values
            return q_root_x, q_root_z, q_root_rot, q_knee, q_hip

        if len(q_values) != 3:
            raise ValueError("The no-platform jumper expects 3 generalized coordinates")
        q_root_rot, q_knee, q_hip = q_values
        return 0.0, 0.0, q_root_rot, q_knee, q_hip

    def center_of_mass_position(self, q_values: tuple[float, float, float, float, float]) -> tuple[float, float]:
        """Return the planar CoM position for one generalized configuration."""

        leg_mass, thigh_mass, trunk_mass = self.segment_masses
        total_mass = leg_mass + thigh_mass + trunk_mass
        segment_centers = self.segment_center_of_mass_positions(q_values)
        leg_com = segment_centers["leg_foot"]
        thigh_com = segment_centers["thigh"]
        trunk_com = segment_centers["trunk"]
        center_of_mass_x = (
            leg_mass * leg_com[0] + thigh_mass * thigh_com[0] + trunk_mass * trunk_com[0]
        ) / total_mass
        center_of_mass_z = (
            leg_mass * leg_com[1] + thigh_mass * thigh_com[1] + trunk_mass * trunk_com[1]
        ) / total_mass
        return center_of_mass_x, center_of_mass_z

    def segment_center_of_mass_positions(
        self,
        q_values: tuple[float, ...] | list[float],
    ) -> dict[str, tuple[float, float]]:
        """Return the planar center of mass of each reduced segment."""

        lengths = self.morphology.segment_lengths
        q_root_x, q_root_z, q_root_rot, q_knee, q_hip = self._unpack_q_values(q_values)
        leg_angle = q_root_rot
        thigh_angle = leg_angle + q_knee
        trunk_angle = thigh_angle + q_hip

        leg_com = (
            q_root_x + 0.55 * lengths.leg_foot * math.sin(leg_angle),
            q_root_z + 0.55 * lengths.leg_foot * math.cos(leg_angle),
        )
        knee = (
            q_root_x + lengths.leg_foot * math.sin(leg_angle),
            q_root_z + lengths.leg_foot * math.cos(leg_angle),
        )
        thigh_com = (
            knee[0] + 0.55 * lengths.thigh * math.sin(thigh_angle),
            knee[1] + 0.55 * lengths.thigh * math.cos(thigh_angle),
        )
        hip = (
            knee[0] + lengths.thigh * math.sin(thigh_angle),
            knee[1] + lengths.thigh * math.cos(thigh_angle),
        )
        trunk_com = (
            hip[0] + 0.5 * lengths.trunk * math.sin(trunk_angle),
            hip[1] + 0.5 * lengths.trunk * math.cos(trunk_angle),
        )
        return {
            "leg_foot": leg_com,
            "thigh": thigh_com,
            "trunk": trunk_com,
        }

    def center_of_mass_horizontal_jacobian(
        self,
        q_values: tuple[float, ...],
    ) -> tuple[float, float, float]:
        """Return the CoM horizontal Jacobian w.r.t. the three rotational DoFs."""

        lengths = self.morphology.segment_lengths
        leg_mass, thigh_mass, trunk_mass = self.segment_masses
        total_mass = leg_mass + thigh_mass + trunk_mass

        _, _, q_root_rot, q_knee, q_hip = self._unpack_q_values(q_values)
        leg_angle = q_root_rot
        thigh_angle = leg_angle + q_knee
        trunk_angle = thigh_angle + q_hip

        d_leg_d_root = 0.55 * lengths.leg_foot * math.cos(leg_angle)
        d_thigh_d_root = lengths.leg_foot * math.cos(leg_angle) + 0.55 * lengths.thigh * math.cos(thigh_angle)
        d_thigh_d_knee = 0.55 * lengths.thigh * math.cos(thigh_angle)
        d_trunk_d_root = (
            lengths.leg_foot * math.cos(leg_angle)
            + lengths.thigh * math.cos(thigh_angle)
            + 0.5 * lengths.trunk * math.cos(trunk_angle)
        )
        d_trunk_d_knee = lengths.thigh * math.cos(thigh_angle) + 0.5 * lengths.trunk * math.cos(trunk_angle)
        d_trunk_d_hip = 0.5 * lengths.trunk * math.cos(trunk_angle)

        jacobian_root = (leg_mass * d_leg_d_root + thigh_mass * d_thigh_d_root + trunk_mass * d_trunk_d_root) / total_mass
        jacobian_knee = (thigh_mass * d_thigh_d_knee + trunk_mass * d_trunk_d_knee) / total_mass
        jacobian_hip = (trunk_mass * d_trunk_d_hip) / total_mass
        return jacobian_root, jacobian_knee, jacobian_hip

    def aligned_initial_joint_configuration_rad(
        self,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 25,
    ) -> tuple[float, ...]:
        """Return one crouched posture with the CoM aligned over the ankle.

        The reduced model has no explicit ankle joint. We therefore use the root
        rotation of the distal segment as one ankle-equivalent rotation and
        converge the horizontal CoM offset toward zero with one Jacobian
        pseudo-inverse update.
        """

        q_values = list(self.crouched_joint_configuration_rad)
        for _ in range(max_iterations):
            center_of_mass_x, _ = self.center_of_mass_position(tuple(q_values))
            ankle_x = q_values[0] if self.floating_base else 0.0
            horizontal_error = center_of_mass_x - ankle_x
            if abs(horizontal_error) <= tolerance:
                break

            jacobian_root, _, _ = self.center_of_mass_horizontal_jacobian(tuple(q_values))
            jacobian_norm_sq = jacobian_root * jacobian_root
            if jacobian_norm_sq <= 1e-12:
                break

            pseudo_inverse = jacobian_root / jacobian_norm_sq
            root_rotation_index = 2 if self.floating_base else 0
            q_values[root_rotation_index] -= pseudo_inverse * horizontal_error

        return tuple(q_values)

    @property
    def initial_joint_configuration_rad(self) -> tuple[float, ...]:
        """Return one nominal crouched initial posture."""

        return self.aligned_initial_joint_configuration_rad()

    def to_biomod_text(self) -> str:
        """Serialize the reduced jumper as plain `bioMod` text."""

        lengths = self.morphology.segment_lengths
        leg_mass, thigh_mass, trunk_mass = self.segment_masses
        leg_inertia = _rod_inertia(leg_mass, lengths.leg_foot)
        thigh_inertia = _rod_inertia(thigh_mass, lengths.thigh)
        trunk_inertia = _rod_inertia(trunk_mass, lengths.trunk)

        return "".join(
            [
                "version 4\n\n",
                f"gravity\t0.0\t0.0\t{-self.gravity:.6f}\n\n",
                _segment_block(
                    "leg_foot",
                    "base",
                    (0.0, 0.0, 0.0),
                    translations="xz" if self.floating_base else None,
                    rotations="y",
                    ranges_q=(
                        (
                            (-1.5, 1.5),
                            (-0.2, 2.2),
                            (-3.141593, 3.141593),
                        )
                        if self.floating_base
                        else ((-3.141593, 3.141593),)
                    ),
                    mass=leg_mass,
                    center_of_mass=(0.0, 0.0, 0.55 * lengths.leg_foot),
                    inertia=leg_inertia,
                ),
                _segment_block(
                    "thigh",
                    "leg_foot",
                    (0.0, 0.0, lengths.leg_foot),
                    rotations="y",
                    ranges_q=((-3.141593, 3.141593),),
                    mass=thigh_mass,
                    center_of_mass=(0.0, 0.0, 0.55 * lengths.thigh),
                    inertia=thigh_inertia,
                ),
                _segment_block(
                    "trunk",
                    "thigh",
                    (0.0, 0.0, lengths.thigh),
                    rotations="y",
                    ranges_q=((-3.141593, 3.141593),),
                    mass=trunk_mass,
                    center_of_mass=(0.0, 0.0, 0.5 * lengths.trunk),
                    inertia=trunk_inertia,
                ),
                _marker_block("foot_contact", "leg_foot", (0.0, 0.0, 0.0)),
                _marker_block("knee", "leg_foot", (0.0, 0.0, lengths.leg_foot)),
                _marker_block("hip", "thigh", (0.0, 0.0, lengths.thigh)),
                _marker_block("head", "trunk", (0.0, 0.0, lengths.trunk)),
                (
                    _contact_block("platform_contact", "leg_foot", (0.0, 0.0, 0.0), axis="z")
                    if self.include_platform_contact
                    else ""
                ),
            ]
        )

    def write_biomod(self, filepath: str | Path) -> Path:
        """Write the reduced model to disk.

        The method prefers `biobuddy` when available and falls back to a plain
        text serializer otherwise.
        """

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._write_biomod_with_biobuddy(output_path)
        except ModuleNotFoundError:
            output_path.write_text(self.to_biomod_text(), encoding="utf-8")
        return output_path

    def _write_biomod_with_biobuddy(self, filepath: Path) -> None:
        """Build and export the reduced model through `biobuddy`."""

        import numpy as np
        from biobuddy import (
            BiomechanicalModelReal,
            ContactReal,
            InertiaParametersReal,
            MarkerReal,
            RangeOfMotion,
            Ranges,
            Rotations,
            RotoTransMatrix,
            SegmentCoordinateSystemReal,
            SegmentReal,
            Translations,
        )

        def scs(translation: tuple[float, float, float]) -> SegmentCoordinateSystemReal:
            rt = RotoTransMatrix()
            rt.from_rotation_matrix_and_translation(np.eye(3), np.array(translation))
            return SegmentCoordinateSystemReal(rt)

        def q_ranges(*bounds: tuple[float, float]) -> RangeOfMotion:
            lower = np.array([bound[0] for bound in bounds], dtype=float)
            upper = np.array([bound[1] for bound in bounds], dtype=float)
            return RangeOfMotion(Ranges.Q, lower, upper)

        lengths = self.morphology.segment_lengths
        leg_mass, thigh_mass, trunk_mass = self.segment_masses

        model = BiomechanicalModelReal(gravity=np.array([0.0, 0.0, -self.gravity]))

        leg_segment = SegmentReal(
            name="leg_foot",
            parent_name="base",
            segment_coordinate_system=scs((0.0, 0.0, 0.0)),
            translations=Translations.XZ if self.floating_base else None,
            rotations=Rotations.Y,
            q_ranges=(
                q_ranges((-1.5, 1.5), (-0.2, 2.2), (-math.pi, math.pi))
                if self.floating_base
                else q_ranges((-math.pi, math.pi))
            ),
            inertia_parameters=InertiaParametersReal(
                mass=leg_mass,
                center_of_mass=np.array([0.0, 0.0, 0.55 * lengths.leg_foot]),
                inertia=np.diag(_rod_inertia(leg_mass, lengths.leg_foot)),
            ),
        )
        leg_segment.add_marker(MarkerReal("foot_contact", position=np.array([0.0, 0.0, 0.0])))
        leg_segment.add_marker(MarkerReal("knee", position=np.array([0.0, 0.0, lengths.leg_foot])))
        if self.include_platform_contact:
            leg_segment.add_contact(ContactReal("platform_contact", position=np.array([0.0, 0.0, 0.0]), axis=Translations.Z))
        model.add_segment(leg_segment)

        thigh_segment = SegmentReal(
            name="thigh",
            parent_name="leg_foot",
            segment_coordinate_system=scs((0.0, 0.0, lengths.leg_foot)),
            rotations=Rotations.Y,
            q_ranges=q_ranges((-math.pi, math.pi),),
            inertia_parameters=InertiaParametersReal(
                mass=thigh_mass,
                center_of_mass=np.array([0.0, 0.0, 0.55 * lengths.thigh]),
                inertia=np.diag(_rod_inertia(thigh_mass, lengths.thigh)),
            ),
        )
        thigh_segment.add_marker(MarkerReal("hip", position=np.array([0.0, 0.0, lengths.thigh])))
        model.add_segment(thigh_segment)

        trunk_segment = SegmentReal(
            name="trunk",
            parent_name="thigh",
            segment_coordinate_system=scs((0.0, 0.0, lengths.thigh)),
            rotations=Rotations.Y,
            q_ranges=q_ranges((-math.pi, math.pi),),
            inertia_parameters=InertiaParametersReal(
                mass=trunk_mass,
                center_of_mass=np.array([0.0, 0.0, 0.5 * lengths.trunk]),
                inertia=np.diag(_rod_inertia(trunk_mass, lengths.trunk)),
            ),
        )
        trunk_segment.add_marker(MarkerReal("head", position=np.array([0.0, 0.0, lengths.trunk])))
        model.add_segment(trunk_segment)

        model.to_biomod(str(filepath), with_mesh=False)
