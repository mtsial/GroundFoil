#!/usr/bin/env python3
"""Zip the samples for a Kaggle dataset: the folders train/ and test/ of data/dataset, each worker
folder with its samples.csv and its not_converged/ samples, and the free-air model
long80k_modelG at the top if it is there (older notebook cells find the data through it).

Before writing, it checks that the samples and the lists of runs agree: every .npz is listed in
the samples.csv of its worker folder and every listed run has its .npz. groundGen.py writes a
sample first and its line second, so a batch that is still running can fail this check: zip only
when all batches have finished. The .npz files are already compressed, so they are stored as
they are. Afterwards every file's checksum is tested and the counts are printed.

Usage (Windows, from the project folder):
    .venv\\Scripts\\python.exe ground_effect\\make_dataset_zip.py data\\kaggle\\ground_effect_final.zip
Only the check: .venv\\Scripts\\python.exe ground_effect\\make_dataset_zip.py --check
"""
import argparse
import csv
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data", "dataset")
MODEL = os.path.join(ROOT, "Deep-Flow-Prediction", "train", "long80k_modelG")
PARTS = ("train", "test")
LISTS = ("samples.csv", "samples2.csv")  # groundGen.py starts samples2.csv after a change of columns


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", nargs="?", help="the zip file to write")
    ap.add_argument("--data", default=DATA, help="folder with train/ and test/ (default: ground_effect/data/dataset)")
    ap.add_argument("--no-model", action="store_true", help="leave long80k_modelG out")
    ap.add_argument("--check", action="store_true", help="only check the samples against their lists, write nothing")
    args = ap.parse_args(argv)
    if not args.check and not args.out:
        ap.error("give the zip file to write, or --check")
    return args


def check(part_dir):
    """Problems found in one part (train or test): .npz files without a line, lines without a file,
    and .npz files outside a worker folder. Returns (problems, number of .npz files)."""
    problems, count, workers = [], 0, set()
    for dirpath, dirnames, files in os.walk(part_dir):
        dirnames.sort()
        if any(name in files for name in LISTS):
            workers.add(dirpath)
    for folder in sorted(workers):
        listed = set()
        for name in LISTS:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                with open(path, newline="") as f:
                    listed |= {row["file"] for row in csv.DictReader(f)}
        found = set()
        for dirpath, _, files in os.walk(folder):
            found |= {os.path.relpath(os.path.join(dirpath, f), folder).replace(os.sep, "/")
                      for f in files if f.endswith(".npz")}
        count += len(found)
        rel = os.path.relpath(folder, part_dir)
        problems += [f"{rel}: {f} has no line in samples.csv" for f in sorted(found - listed)]
        problems += [f"{rel}: {f} is listed but missing" for f in sorted(listed - found)]
    for dirpath, _, files in os.walk(part_dir):
        if not any(dirpath == w or dirpath.startswith(w + os.sep) for w in workers):
            problems += [f"{os.path.relpath(os.path.join(dirpath, f), part_dir)}: outside a worker folder"
                         for f in files if f.endswith(".npz")]
    return problems, count


def main(argv=None):
    args = parse(argv)
    problems = []
    for part in PARTS:
        if not os.path.isdir(os.path.join(args.data, part)):
            sys.exit(f"{os.path.join(args.data, part)} not found")
        found, n = check(os.path.join(args.data, part))
        print(f"{part}: {n} samples, {len(found)} problems")
        problems += [f"{part}/{p}" for p in found]
    if problems:
        print("\n".join(problems[:20]) + (f"\n... and {len(problems) - 20} more" if len(problems) > 20 else ""))
        sys.exit("the samples and their lists do not agree (is a batch still running?)"
                 + ("" if args.check else ": nothing written"))
    if args.check:
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for part in PARTS:
            for dirpath, dirnames, files in os.walk(os.path.join(args.data, part)):
                dirnames.sort()
                for f in sorted(files):
                    if f.endswith((".npz", ".csv")):
                        full = os.path.join(dirpath, f)
                        z.write(full, os.path.relpath(full, args.data).replace(os.sep, "/"))
        if not args.no_model and os.path.isfile(MODEL):
            z.write(MODEL, "long80k_modelG")

    with zipfile.ZipFile(args.out) as z:
        bad = z.testzip()
        names = z.namelist()
    print(f"written {args.out}: {len(names)} files, {os.path.getsize(args.out) / 1e6:.0f} MB; "
          f"checksums {'OK' if bad is None else 'FAILED at ' + bad}")
    for part in PARTS:
        npz = [n for n in names if n.startswith(part + "/") and n.endswith(".npz")]
        bad_runs = sum("/not_converged/" in n for n in npz)
        lists = sum(n.startswith(part + "/") and n.endswith(".csv") for n in names)
        print(f"  {part}: {len(npz) - bad_runs} converged samples, {bad_runs} not converged, {lists} lists of runs")
    print(f"  long80k_modelG: {'yes' if 'long80k_modelG' in names else 'no'}")
    if bad is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
