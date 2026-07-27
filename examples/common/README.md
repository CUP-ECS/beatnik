# examples/common

Shared helpers for the example drivers under `examples/NN_name/`.

Anything that more than one example needs — deck parsing, output-path helpers,
banner printing, common CLI handling — belongs here rather than being copied
between example directories. Each example directory holds **one** driver source
plus its own `CMakeLists.txt`; everything shared moves up to here.

Empty today. The first candidate for promotion is
[../01_rising_bubble/InputFile.hpp](../01_rising_bubble/InputFile.hpp), the
`key = value` deck parser — move it here once a second example needs it, and
mind that it currently targets the pre-redesign solver's parameter struct.

New sources here carry the project's BSD-3-Clause header block, like every other
source file in the repository.
