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

    @property
    def q_size(self) -> int:
        """Return the number of generalized coordinates."""

        return 5

    @property
    def tau_size(self) -> int:
        """Return the number of actuated generalized torques."""

        return 2

    @property
    def segment_masses(self) -> tuple[float, float, float]:
        """Return the reduced mass allocation across the three segments."""

        total = self.morphology.mass_kg
        return (0.12 * total, 0.24 * total, 0.64 * total)

    @property
    def initial_joint_configuration_rad(self) -> tuple[float, float, float, float, float]:
        """Return one nominal crouched initial posture."""

        flexion = math.radians(self.morphology.initial_joint_flexion_deg)
        return (0.0, 0.0, 0.0, -flexion, flexion)

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
                    translations="xz",
                    rotations="y",
                    ranges_q=(
                        (-1.5, 1.5),
                        (-0.2, 2.2),
                        (-3.141593, 3.141593),
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
                    center_of_mass=(0.0, 0.0, 0.45 * lengths.thigh),
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
                _contact_block("platform_contact", "leg_foot", (0.0, 0.0, 0.0), axis="z"),
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
            translations=Translations.XZ,
            rotations=Rotations.Y,
            q_ranges=q_ranges((-1.5, 1.5), (-0.2, 2.2), (-math.pi, math.pi)),
            inertia_parameters=InertiaParametersReal(
                mass=leg_mass,
                center_of_mass=np.array([0.0, 0.0, 0.55 * lengths.leg_foot]),
                inertia=np.diag(_rod_inertia(leg_mass, lengths.leg_foot)),
            ),
        )
        leg_segment.add_marker(MarkerReal("foot_contact", position=np.array([0.0, 0.0, 0.0])))
        leg_segment.add_marker(MarkerReal("knee", position=np.array([0.0, 0.0, lengths.leg_foot])))
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
                center_of_mass=np.array([0.0, 0.0, 0.45 * lengths.thigh]),
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
