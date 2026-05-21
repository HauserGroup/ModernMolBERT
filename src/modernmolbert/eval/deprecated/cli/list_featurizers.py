from modernmolbert.eval.registry import list_featurizers


def main() -> None:
    for item in list_featurizers():
        extra = item["required_extra"] or "core"
        print(f"{item['name']}\t[{extra}]\t{item['description']}")


if __name__ == "__main__":
    main()
