import sastvd.helpers.datasets as svdd
import sastvd.ivdetect.evaluate as ivde
import pdb


def bigvul():
    """Run preperation scripts for BigVul dataset."""
    # df = svdd.bigvul()
    # df.to_csv("bigvul.csv", index=False)
    # pdb.set_trace()
    # lines = ivde.get_dep_add_lines_bigvul()
    # print(lines)
    # pdb.set_trace()
    df = svdd.bigvul()
    # import pdb
    # pdb.set_trace()
    ivde.get_dep_add_lines_bigvul()
    # svdd.generate_glove("bigvul")
    # svdd.generate_d2v("bigvul")


if __name__ == "__main__":
    bigvul()
