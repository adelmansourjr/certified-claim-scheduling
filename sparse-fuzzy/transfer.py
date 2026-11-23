"""Apply the frozen WiCE execution policies to a pinned SciFact workload."""

import argparse
import importlib.util
import json
from pathlib import Path
import tarfile

SOURCES = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("sparse_wice", SOURCES/"study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)

from study import Case


ARTICLE_DIGEST = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
SOURCE = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def read_cases(source):
    parent = study.live
    assert parent.file_sha(source)==ARTICLE_DIGEST
    with tarfile.open(source,"r:gz") as archive:
        def load(name):
            item=archive.getmember(f"data/{name}.jsonl")
            assert item.isfile()
            with archive.extractfile(item) as handle:
                return [json.loads(row) for row in handle if row.strip()]
        claim_rows=load("claims_dev")
        articles={int(row["doc_id"]):row for row in load("corpus")}

    cases=[]
    for row in sorted(claim_rows,key=lambda r:r["id"]):
        for doc_id in sorted(set(map(int,row["cited_doc_ids"]))):
            article=articles[doc_id]
            title=" ".join(parent.words(article["title"]))
            lines=[article["title"],*article["abstract"]]
            # Dummy labels are storage placeholders. The output omits quality statistics.
            cases.append(Case(
                f'{row["id"]}/{doc_id}',title,str(doc_id),str(row["id"]),lines,[row["claim"]],(0,)
            ))
    assert len(cases)==len({row.key for row in cases})
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scifact",type=Path)
    parser.add_argument("--data",type=Path,default=study.PARENT/"data")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    lock=json.loads(study.LOCK.read_text())
    assert study.live.file_sha(study.ARCHIVE)==lock["model"]
    assert all(study.live.file_sha(study.ROOT/f)==h for f,h in lock["source"].items())
    train_path=args.data/"train.jsonl"
    assert study.live.file_sha(train_path)==lock["data"]["train"]
    train=study.live.load(train_path)
    idf,unknown=study.live.build_idf(train)
    fit=json.loads(study.ARCHIVE.read_text())
    cases=read_cases(args.scifact)
    rows=study.evaluate(cases,idf,unknown,fit,5)
    candidate=lock["candidate"]
    summary={
        "parents":len(rows),
        "baseline":lock["baseline"],
        "candidate":candidate,
        "sum_case_median_cpu_ns":{method:study.total_cpu(rows,method) for method in study.METHODS},
        "contrasts":{method:study.live.uncertainty(rows,"times",method,candidate)
                     for method in (*study.CONTROLS,candidate+"_eager")},
    }
    summary["cohort"]="SciFact development, all released claim-cited-abstract pairs"
    summary["claims"]=len({case.context for case in cases})
    summary["abstracts"]=len({case.doc_id for case in cases})
    for row in rows:
        del row["labels"]
    study.save(args.out,rows,summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
