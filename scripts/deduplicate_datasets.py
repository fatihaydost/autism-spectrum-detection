import json
import shutil
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORT_PATH = DATA_DIR / "reports" / "combined_dataset_duplicates_report.json"
DUPLICATE_ARCHIVE_DIR = DATA_DIR / "archives" / "duplicates_removed"

DATASET_LOCATION_HINTS: dict[str, Path] = {
    "ASD Data": DATA_DIR / "processed" / "asd_faces",
    "Autistic Children": DATA_DIR
    / "raw_sources"
    / "Autistic Children Facial Image Dataset",
    "Autistic Children Facial Image Dataset": DATA_DIR
    / "raw_sources"
    / "Autistic Children Facial Image Dataset",
    "Facial Autistic Children": DATA_DIR
    / "raw_sources"
    / "Facial dataset of autistic children",
    "Facial dataset of autistic children": DATA_DIR
    / "raw_sources"
    / "Facial dataset of autistic children",
    "old_dataset": DATA_DIR / "raw_sources" / "old_dataset",
}

# Prefer keeping images from the cleaner splits first.
DATASET_PRIORITY = [
    "ASD Data",
    "Autistic Children",
    "Facial Autistic Children",
    "old_dataset",
]
PRIORITY_INDEX = {name: index for index, name in enumerate(DATASET_PRIORITY)}


def load_report() -> dict:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Report not found: {REPORT_PATH}")

    with REPORT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_image_path(entry: dict, mapping: dict) -> tuple[Path | None, bool]:
    """
    Turn an entry in the duplicate report into an absolute path on disk.

    The input datasets were collected on different machines, so the report
    stores relative paths that are not perfectly consistent. We probe a few
    possible layouts and return the first one that exists.
    """

    relative = Path(entry["relative_path"])
    dataset = entry["dataset"]
    mapping_value = mapping.get(dataset, dataset)
    dataset_root = ROOT_DIR / Path(mapping_value)

    hinted_root = DATASET_LOCATION_HINTS.get(mapping_value)
    if hinted_root is not None:
        dataset_root = hinted_root

    candidates = [
        dataset_root / relative,
        dataset_root / dataset / relative,
    ]

    if len(relative.parts) > 1:
        tail = Path(*relative.parts[1:])
        candidates.extend(
            [
                dataset_root / tail,
                dataset_root / dataset / tail,
            ]
        )

    # Also try the naive root-relative path as a final fallback.
    candidates.append(ROOT_DIR / relative)

    probed = set()
    archive_hit = False

    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except FileNotFoundError:
            # On Windows resolve() may raise if the path does not exist.
            candidate = candidate

        # Avoid probing the same candidate twice.
        if candidate in probed:
            continue
        probed.add(candidate)

        if candidate.exists():
            return candidate, False

        relative_candidate = None
        for base in (DATA_DIR, ROOT_DIR):
            try:
                relative_candidate = candidate.relative_to(base)
                break
            except ValueError:
                continue

        if (
            relative_candidate is not None
            and (DUPLICATE_ARCHIVE_DIR / relative_candidate).exists()
        ):
            archive_hit = True

    return None, archive_hit


def iter_duplicate_groups(report: dict):
    yield from report.get("exact_matches", [])
    yield from report.get("visual_matches", [])


def main() -> None:
    report = load_report()
    mapping = report["metadata"]["dataset_mapping"]

    DUPLICATE_ARCHIVE_DIR.mkdir(exist_ok=True)

    moved_paths = set()
    moved_per_dataset = defaultdict(int)
    unresolved_entries = []

    total_groups = 0
    processed_groups = 0

    for group in iter_duplicate_groups(report):
        total_groups += 1

        images = group.get("images", [])
        if len(images) < 2:
            continue

        resolved = []
        for entry in images:
            path, archived = resolve_image_path(entry, mapping)
            if path is None:
                if archived:
                    continue
                unresolved_entries.append(
                    f"Unresolved path for dataset={entry['dataset']} "
                    f"relative_path={entry['relative_path']}"
                )
                continue
            resolved.append((path, entry))

        resolved = [
            item
            for item in resolved
            if item[0] not in moved_paths and item[0].exists()
        ]

        if not resolved:
            continue

        resolved.sort(
            key=lambda pair: (
                PRIORITY_INDEX.get(pair[1]["dataset"], len(PRIORITY_INDEX)),
                str(pair[0]),
            )
        )

        keeper_path, _ = resolved[0]
        processed_groups += 1

        for candidate_path, entry in resolved[1:]:
            if candidate_path in moved_paths or not candidate_path.exists():
                continue

            archive_relative = None
            for base in (DATA_DIR, ROOT_DIR):
                try:
                    archive_relative = candidate_path.relative_to(base)
                    break
                except ValueError:
                    continue
            if archive_relative is None:
                archive_relative = candidate_path.name

            archive_path = DUPLICATE_ARCHIVE_DIR / archive_relative
            archive_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(candidate_path), str(archive_path))
            moved_paths.add(candidate_path)
            moved_per_dataset[entry["dataset"]] += 1

    moved_total = sum(moved_per_dataset.values())

    print(f"Duplicate groups found: {total_groups}")
    print(f"Groups processed      : {processed_groups}")
    print(f"Images moved          : {moved_total}")
    if moved_total:
        print("Breakdown by dataset:")
        for dataset, count in sorted(moved_per_dataset.items()):
            print(f"  - {dataset}: {count}")

    if unresolved_entries:
        print("\nEntries that could not be resolved on disk:")
        for line in unresolved_entries[:10]:
            print(f"  {line}")
        if len(unresolved_entries) > 10:
            print(f"  ... {len(unresolved_entries) - 10} more")


if __name__ == "__main__":
    main()
