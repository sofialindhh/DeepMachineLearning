"""Register the local architecture copy under the name used by the v2.6 checkpoint."""

from tabpfn.architectures import ARCHITECTURES

import my_arch

OFFICIAL_ARCH_NAME = "tabpfn_v2_6"


def use_as_v26() -> None:
    """Load the official v2.6 checkpoint into ``my_arch``.

    TabPFN selects the architecture by the ``architecture_name`` stored in the
    checkpoint. For v2.6 that name is ``"tabpfn_v2_6"``, so the registry entry
    must point to this local module.
    """
    ARCHITECTURES[OFFICIAL_ARCH_NAME] = my_arch
