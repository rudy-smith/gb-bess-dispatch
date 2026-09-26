"""Report the field definitions and sample rows of each NESO resource.
 
The normalisation layer has to map NESO's column names onto the settlement
date, settlement period and value that the rest of the pipeline expects. Those
names are not documented uniformly across datasets and have changed before, so
they are read from the API rather than assumed.
 
This script makes no assumptions and writes nothing. It prints what the API
says each resource contains, which is the input to pinning the mapping.
 
Usage:
 
    python -m scripts.probe_neso
    python -m scripts.probe_neso --resource wind_da --rows 5
"""
 
from __future__ import annotations
 
import argparse
import json
import logging
 
from src.fetch.neso import RESOURCES, describe_resource
 
 
def report(name: str, rows: int) -> None:
    resource = RESOURCES[name]
    print(f"\n=== {name}: {resource.description} ===")
    print(f"resource_id              {resource.resource_id}")
    print(
        f"assumed publication      {resource.publication_local_time} local, "
        f"{resource.lead_days} day(s) before target"
    )
 
    try:
        described = describe_resource(resource.resource_id, sample_rows=rows)
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        return
 
    print(f"total rows               {described['total']:,}")
    print("\nfields:")
    for field in described["fields"]:
        print(f"  {field.get('id'):<32} {field.get('type')}")
 
    print("\nsample rows:")
    for record in described["records"]:
        print("  " + json.dumps(record, default=str))
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resource",
        choices=sorted(RESOURCES),
        default=None,
        help="probe one resource; omit to probe all",
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
 
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
 
    names = [args.resource] if args.resource else sorted(RESOURCES)
    for name in names:
        report(name, args.rows)
 
    print(
        "\nNext: the settlement date, settlement period and forecast value columns "
        "above are what the normaliser maps onto. Report them before it is written."
    )
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())

 