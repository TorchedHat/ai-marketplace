"""Structured output objects for the vLLM test oracle auditor."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class AuditCandidate:
    """Phase 1 evidence for a single test candidate."""

    candidate: str
    file: str
    line: int
    comparison: str
    oracle: str
    helper: str
    batch_invariant_enabled: bool
    code_path_verified: bool
    fixtures: str
    c1_weak_oracle: str
    c2_realistic_breakage: str
    c3_no_strong_contract: str
    classification: str
    verdict: str
    code_snippet: str = ""


@dataclass
class ReviewCandidate:
    """Phase 2 review verdict for a single test candidate."""

    candidate: str
    phase_1_classification: str
    phase_1_verdict: str
    review: str
    file: str
    line: int
    comparison: str
    oracle: str
    helper: str
    batch_invariant_enabled: bool
    code_path_verified: bool
    fixtures: str
    c1_weak_oracle: str
    c2_realistic_breakage: str
    c3_no_strong_contract: str
    classification: str
    verdict: str
    code_snippet: str = ""


@dataclass
class AuditReport:
    """Full Phase 1 audit report."""

    test_files_in_scope: int
    candidates_analyzed: int
    candidates: list[AuditCandidate] = field(default_factory=list)

    def verify_coverage(self, expected_tests: list[str]) -> None:
        """Assert every test from list_tests.sh output has a candidate entry.

        Args:
            expected_tests: list of "DIR,FILE,FUNCTION" csv lines from list_tests.sh
        """
        expected = set()
        for line in expected_tests:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 3:
                expected.add(parts[2])
        analyzed = {c.candidate for c in self.candidates}
        missing = expected - analyzed
        if missing:
            raise AssertionError(
                f"Missing {len(missing)} test(s) from report:\n"
                + "\n".join(sorted(missing))
            )

    def write_to_file(self, file_name: str) -> None:
        """Write report as JSON to the given file path."""
        Path(file_name).write_text(
            json.dumps(asdict(self), indent=2) + "\n"
        )


@dataclass
class ReviewReport:
    """Full Phase 2 review report."""

    test_files_in_scope: int
    candidates_analyzed: int
    phase_1_agreed: int
    phase_1_reclassified: int
    candidates: list[ReviewCandidate] = field(default_factory=list)

    def write_to_file(self, file_name: str) -> None:
        """Write report as JSON to the given file path."""
        Path(file_name).write_text(
            json.dumps(asdict(self), indent=2) + "\n"
        )
