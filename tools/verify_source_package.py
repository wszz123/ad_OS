"""Verify the source manifest and smoke-test an extracted package independently."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    output = Path(args.output_directory).resolve()
    with zipfile.ZipFile(args.archive) as archive:
        manifest = json.loads(archive.read("PACKAGE_MANIFEST.json"))
        allowed = {item["path"] for item in manifest["files"]} | {"PACKAGE_MANIFEST.json"}
        if set(archive.namelist()) != allowed:
            raise ValueError("ZIP members do not match manifest")
        for item in manifest["files"]:
            path = PurePosixPath(item["path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("package contains an invalid member path")
            content = archive.read(item["path"])
            if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError(f"package checksum failed: {item['path']}")
        output.mkdir(parents=True, exist_ok=False)
        for name in allowed:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
    # The package's own modules resolve from the extracted cwd, not the original repo.
    command = [sys.executable, "-m", "coop", "demo", "--backend", "fixture", "--skip-recovery",
               "--output-root", str(output / "results" / "package_smoke")]
    result = subprocess.run(command, cwd=output, capture_output=True, text=True, timeout=180)
    report = {"manifest_verified": True, "file_count": len(manifest["files"]), "extracted_directory": str(output),
              "fixture_smoke_return_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
              "scope": "Extracted package under the current environment; not a fresh-environment install or learned-model run"}
    (output / "PACKAGE_VERIFICATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("stdout", "stderr")}, ensure_ascii=False, indent=2))
    if result.returncode:
        raise SystemExit("extracted package smoke failed; see PACKAGE_VERIFICATION.json")


if __name__ == "__main__":
    main()
