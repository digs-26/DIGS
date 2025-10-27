import os
import pdb
import re

import pandas as pd
import sastvd as svd
import sastvd.helpers.doc2vec as svdd2v
import sastvd.helpers.git as svdg
import sastvd.helpers.glove as svdglove
import sastvd.helpers.tokenise as svdt
from sklearn.model_selection import train_test_split
import importlib
import config
import numpy as np

def train_val_test_split_df(df, idcol, labelcol):
    """Add train/val/test column into dataframe."""
    X = df[idcol]
    y = df[labelcol]
    train_rat = 0.8
    val_rat = 0.1
    test_rat = 0.1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_rat, random_state=1
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_rat / (test_rat + val_rat), random_state=1
    )
    X_train = set(X_train)
    X_val = set(X_val)
    X_test = set(X_test)

    def path_to_label(path):
        if path in X_train:
            return "train"
        if path in X_val:
            return "val"
        if path in X_test:
            return "test"

    df["label"] = df[idcol].apply(path_to_label)
    return df


def remove_comments(text):
    """Delete comments from code."""
    
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s
    try:
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE,
        )
        return re.sub(pattern, replacer, text)
    except:
        return ""


def generate_glove(dataset="bigvul", sample=False, cache=True):
    """Generate Glove embeddings for tokenised dataset."""
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"glove_{sample}")
    if os.path.exists(savedir / "vectors.txt") and cache:
        svd.debug("Already trained GloVe.")
        return
    if dataset == "bigvul":
        df = bigvul(sample=sample)
    MAX_ITER = 2 if sample else 500

    # Only train GloVe embeddings on train samples
    samples = df[df.label == "train"].copy()

    # Preprocessing
    samples.before = svd.dfmp(
        samples, svdt.tokenise_lines, "before", cs=200, desc="Get lines: "
    )
    lines = [i for j in samples.before.to_numpy() for i in j]

    # Save corpus
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"glove_{sample}")
    with open(savedir / "corpus.txt", "w") as f:
        f.write("\n".join(lines))

    # Train Glove Model
    CORPUS = savedir / "corpus.txt"
    svdglove.glove(CORPUS, MAX_ITER=MAX_ITER)


def generate_d2v(dataset="bigvul", sample=False, cache=True, **kwargs):
    """Train Doc2Vec model for tokenised dataset."""
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"d2v_{sample}")
    if os.path.exists(savedir / "d2v.model") and cache:
        svd.debug("Already trained Doc2Vec.")
        return
    if dataset == "bigvul":
        df = bigvul(sample=sample)

    # Only train Doc2Vec on train samples
    samples = df[df.label == "train"].copy()

    # Preprocessing
    samples.before = svd.dfmp(
        samples, svdt.tokenise_lines, "before", cs=200, desc="Get lines: "
    )
    lines = [i for j in samples.before.to_numpy() for i in j]

    # Train Doc2Vec model
    model = svdd2v.train_d2v(lines, **kwargs)

    # Test Most Similar
    most_sim = model.dv.most_similar([model.infer_vector("memcpy".split())])
    for i in most_sim:
        print(lines[i[0]])
    model.save(str(savedir / "d2v.model"))


# def bigvul(minimal=False, sample=False, return_raw=False, splits="default"):
#     """Read BigVul Data.
#
#     Args:
#         sample (bool): Only used for testing!
#         splits (str): default, crossproject-(linux|Chrome|Android|qemu)
#
#     EDGE CASE FIXING:
#     id = 177860 should not have comments in the before/after
#     """
#     # savedir = svd.get_dir(svd.cache_dir() / "minimal_datasets")
#     filename = "MSR_data_cleaned_SAMPLE.csv" if sample else "MSR_data_cleaned.csv"
#     df = pd.read_csv(svd.external_dir() / filename)
#     df = df.rename(columns={"Unnamed: 0": "id"})
#     df["dataset"] = "bigvul"
#
#     # Remove comments
#     df["func_before"] = svd.dfmp(df, remove_comments, "func_before", cs=500)
#     df["func_after"] = svd.dfmp(df, remove_comments, "func_after", cs=500)
#
#     # Save codediffs
#     cols = ["func_before", "func_after", "id", "dataset"]
#     svd.dfmp(df, svdg._c2dhelper, columns=cols, ordr=False, cs=300)
#
#     # Assign info and save
#     df["info"] = svd.dfmp(df, svdg.allfunc, cs=500)
#     df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)
#
#     # POST PROCESSING
#     dfv = df[df.vul == 1]
#     # No added or removed but vulnerable
#     dfv = dfv[~dfv.apply(lambda x: len(x.added) == 0 and len(x.removed) == 0, axis=1)]
#     # Remove functions with abnormal ending (no } or ;)
#     try:
#         dfv = dfv[
#             ~dfv.apply(
#                 lambda x: x.func_before.strip()[-1] != "}"
#                 and x.func_before.strip()[-1] != ";",
#                 axis=1,
#             )
#         ]
#         dfv = dfv[
#             ~dfv.apply(
#                 lambda x: x.func_after.strip()[-1] != "}" and x.after.strip()[-1:] != ";",
#                 axis=1,
#             )
#         ]
#     except:
#         pass
#
#     # Remove functions with abnormal ending (ending with ");")
#     dfv = dfv[~dfv.before.apply(lambda x: x[-2:] == ");")]
#
#     # Remove samples with mod_prop > 0.5
#     dfv["mod_prop"] = dfv.apply(
#         lambda x: len(x.added + x.removed) / len(x["diff"].splitlines()), axis=1
#     )
#     dfv = dfv.sort_values("mod_prop", ascending=0)
#     dfv = dfv[dfv.mod_prop < 0.7]
#     # Remove functions that are too short
#     dfv = dfv[dfv.apply(lambda x: len(x.before.splitlines()) > 5, axis=1)]
#     # Filter by post-processing filtering
#     keep_vuln = set(dfv.id.tolist())
#     df = df[(df.vul == 0) | (df.id.isin(keep_vuln))].copy()
#
#     # Make splits
#     df = train_val_test_split_df(df, "id", "vul")
#
#     metadata_cols = df.columns[:17].tolist() + ["project"]
#     df[metadata_cols].to_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv", index=0)
#     return df
def process_row(row):
    vul_labels = row['removed']
    lines = row['before'].splitlines()
    vul_pattern = ""
    vul_lines = []

    if not vul_labels:
        fix_labels = row['added']
        if not fix_labels:
            row['vul_patterns'] = "na"
            row['statement_label'] = [0] * 155
            return row
        line = lines[fix_labels[0] - 1]
        vul_pattern += line + '<SPLIT>'
        vul_lines.append(fix_labels[0] - 1)
    else:
        for vul_label in vul_labels:
            if vul_label < len(lines):
                line = lines[vul_label]
                vul_pattern += line + '<SPLIT>'
                vul_lines.append(vul_label)

    row['vul_patterns'] = vul_pattern
    statement_label = [0] * 155
    for vul_line in vul_lines:
        if vul_line < 155:
            statement_label[vul_line] = 1
    row['statement_label'] = statement_label

    return row

def tocsv(df):
    output = df["dataset"].iloc[0] + '.csv'
    df = df.rename(columns={"id": "index", "func_after": "vul_func_with_fix", "vul": "function_label"})
    df = df.apply(process_row, axis=1)
    df = df[['index', 'vul_func_with_fix', 'func_before', 'vul_patterns', 'function_label', 'statement_label']]
    df.to_csv(output, index=False)
# def tocsv(df):
#     df = df.rename(columns={"id": "index"})
#     df = df.rename(columns={"func_after": "vul_func_with_fix"})
#     df = df.rename(columns={"vul": "function_label"})
#     output = df['project'].iloc[0] + '.csv'
#     for i in range(len(df)):
#         vul_labels = df.at[i, 'removed']
#         lines = df.at[i, 'before'].splitlines()
#         vul_pattern = ""
#         vul_lines = []
#
#         if not vul_labels:
#             fix_labels = df.at[i, 'added']
#             if not fix_labels:
#                 df.at[i, 'vul_pattern'] = "na"
#                 continue
#             line = lines[fix_labels[0] - 1]
#             vul_pattern += line + '<SPLIT>'
#             vul_lines.append(fix_labels[0] - 1)
#         else:
#             for vul_label in vul_labels:
#                 line = lines[vul_label]
#                 vul_pattern += line + '<SPLIT>'
#                 vul_lines.append(vul_label)
#
#         df.at[i, 'vul_pattern'] = vul_pattern
#         statement_label = [0] * 155
#         for vul_line in vul_lines:
#             statement_label[vul_line] = 1
#         df.at[i, 'statement_label'] = statement_label
#
#     df = df[['index', 'vul_func_with_fix', 'vul_before', 'vul_pattern', 'function_label', 'statement_label']]
#     df.to_csv(output, index=False)


def bigvul(minimal=False, sample=False, return_raw=False, splits="default"):
    df_train = process_and_save(config.get_trainset(), part=False)
    df_part = process_and_save(config.get_partset(), part=True)
    df_test = process_and_save(config.get_testset(), part=False)

    df_train["label"] = "train"
    df_part["label"] = "train"
    df_test["label"] = "test"

    # tocsv(df_train)
    # tocsv(df_part)
    # # tocsv(df_test)
    # import pdb
    # pdb.set_trace()

    df_all = pd.concat([df_train, df_part, df_test], axis=0, ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=config.get_seed()).reset_index(drop=True)
    return df_all


def process_and_save(project, part=False):
    filename = f'{project}.csv'
    df = pd.read_csv(svd.external_dir() / filename)

    df["dataset"] = project
    if project == 'bigvul':
        df = df.rename(columns={"Unnamed: 0": "id"})
    else:
        df['id'] = df['id'].astype(str) + '_' + df['dataset']

    # Remove comments
    df["func_before"] = svd.dfmp(df, remove_comments, "func_before", cs=500)
    df["func_after"] = svd.dfmp(df, remove_comments, "func_after", cs=500)

    # Save codediffs
    cols = ["func_before", "func_after", "id", "dataset"]
    svd.dfmp(df, svdg._c2dhelper, columns=cols, ordr=False, cs=300)

    # Assign info and save
    df["info"] = svd.dfmp(df, svdg.allfunc, cs=500)
    df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)

    # POST PROCESSING
    dfv = df[df.vul == 1]
    # print(f'{project} original vul length: {len(dfv)}')
    # No added or removed but vulnerable
    dfv = dfv[~dfv.apply(lambda x: len(x.added) == 0 and len(x.removed) == 0, axis=1)]
    # Remove functions with abnormal ending (no } or ;)
    try:
        dfv = dfv[
            ~dfv.apply(
                lambda x: x.func_before.strip()[-1] != "}"
                          and x.func_before.strip()[-1] != ";",
                axis=1,
            )
        ]
        dfv = dfv[
            ~dfv.apply(
                lambda x: x.func_after.strip()[-1] != "}" and x.after.strip()[-1:] != ";",
                axis=1,
            )
        ]
    except:
        pass

    # Remove functions with abnormal ending (ending with ");")
    dfv = dfv[~dfv.before.apply(lambda x: x[-2:] == ");")]
    # print(f'{project} abnormal ending length: {len(dfv)}')
    # Remove samples with mod_prop > 0.7
    dfv["mod_prop"] = dfv.apply(
        lambda x: len(x.added + x.removed) / len(x["diff"].splitlines()), axis=1
    )
    dfv = dfv.sort_values("mod_prop", ascending=0)
    dfv = dfv[dfv.mod_prop < 0.7]
    # print(f'{project} mod_prop length: {len(dfv)}')
    # Remove functions that are too short
    dfv = dfv[dfv.apply(lambda x: len(x.before.splitlines()) > 5, axis=1)]
    # print(f'{project} too short length: {len(dfv)}')
    # Filter by post-processing filtering
    keep_vuln = set(dfv.id.tolist())
    df = df[(df.vul == 0) | (df.id.isin(keep_vuln))].copy()

    if part:
        print(f'partset {project} length: {len(df)}')
        if config.get_selection() == 'random':
            under = config.get_under()
            if under != 1.0:
                seed = config.get_seed()
                print(f'partset {project} under random: {under}')
                df = df.sample(frac=under, random_state=seed)
            else:
                print(f'partset {project} under random: {under}')
        else:
            indices_path = config.get_indices_path()
            indices = np.loadtxt(indices_path, dtype=str)
            print(f'partset {project} indices length: {len(indices)}')

            ids = []
            for i in range(len(indices)):
                ids.append(indices[i] + '_' + project)
            df = df[df.id.isin(ids)]
        print(f'partset {project} processed length: {len(df)}')
    print(f'{project} processed length: {len(df)}')

    return df

def bigvul_cve():
    """Return id to cve map."""
    md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
    ret = md[["id", "CVE ID"]]
    return ret.set_index("id").to_dict()["CVE ID"]
