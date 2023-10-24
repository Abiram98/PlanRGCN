from feature_extraction.predicates.pred_util import *


class PredicateFreqExtractor(ExtractorBase):
    def __init__(
        self,
        endpoint: Endpoint,
        input_dir: str,
        output_dir: str,
        predicate_file="predicates.json",
    ) -> None:
        super().__init__(endpoint, output_dir, predicate_file)
        self.input_dir = input_dir
        self.predicates = json.load(
            open(f"{self.input_dir}/predicate/{self.predicate_file}", "r")
        )
        self.batch_output_dir = f"{output_dir}/predicate/pred_stat/batches"
        os.system(f"mkdir -p {self.batch_output_dir}")
        # the
        # path to where the responses are saved with the features.
        self.batch_output_response_dir = (
            f"{output_dir}/predicate/pred_stat/batches_response_stats"
        )
        os.system(f"mkdir -p {self.batch_output_response_dir}")
        os.system(f"mkdir -p {self.batch_output_response_dir}/freq")
        os.system(f"mkdir -p {self.batch_output_response_dir}/ents")
        os.system(f"mkdir -p {self.batch_output_response_dir}/lits")
        os.system(f"mkdir -p {self.batch_output_response_dir}/subj")
        os.system(f"mkdir -p {self.batch_output_response_dir}/obj")
        self.load_batches()

    def query_batches(self, batch_start=1, batch_end=2):
        if not hasattr(self, "batches"):
            self.load_batches()
        save_path = self.batch_output_response_dir
        os.system(f"mkdir -p {save_path}")
        print(f"Predicate Stats are saved to: {save_path}")
        batch_end_idx = min(batch_end - 1, len(self.batches) - 1)
        if batch_end == -1:
            batch_end_idx = len(self.batches) - 1
        for i, b in enumerate(self.batches[batch_start - 1 : batch_end_idx]):
            for query_generator, name in zip(
                [
                    PredicateStatQueries.unique_literals_for_predicate,
                    PredicateStatQueries.unique_entity_for_predicate,
                    PredicateStatQueries.frequency_for_predicate,
                ],
                ["lits", "ents", "freq"],
            ):
                query = query_generator(b)
                res = self.endpoint.run_query(query)
                res_fp = f"{save_path}/{name}/batch_{batch_start+i}.json"
                json.dump(res, open(res_fp, "w"))
                print(f"batch {batch_start+i} extracted!")

    def query_batches_subj_obj(self, batch_start=1, batch_end=2):
        if not hasattr(self, "batches"):
            self.load_batches()
        save_path = self.batch_output_response_dir
        os.system(f"mkdir -p {save_path}")
        print(f"Predicate Stats are saved to: {save_path}")
        batch_end_idx = min(batch_end - 1, len(self.batches) - 1)
        if batch_end == -1:
            batch_end_idx = len(self.batches) - 1
        for i, b in enumerate(self.batches[batch_start - 1 : batch_end_idx]):
            for query_generator, name in zip(
                [
                    PredicateStatQueries.unique_subj_for_predicate,
                    PredicateStatQueries.unique_obj_for_predicate,
                ],
                ["subj", "obj"],
            ):
                query = query_generator(b)
                res = self.endpoint.run_query(query)
                res_fp = f"{save_path}/{name}/batch_{batch_start+i}.json"
                json.dump(res, open(res_fp, "w"))
                print(f"batch {batch_start+i} extracted!")


class PredicateStatQueries:
    def pred_str_gen(batch):
        pred_str = ""
        for p in batch:
            if " " in p:
                raise Exception("space in pred")
            pred_str += f"(<{p}>) "
        return pred_str

    def unique_literals_for_predicate(batch):
        pred_str = PredicateStatQueries.pred_str_gen(batch)
        return f"""SELECT ?p1 (COUNT(DISTINCT ?o) AS ?literals) WHERE {{
            VALUES (?p1) {{ {pred_str}}}
            ?s ?p1 ?o .
            FILTER(isLiteral(?o))
        }}
        GROUP BY ?p1
        """

    def unique_entity_for_predicate(batch):
        pred_str = PredicateStatQueries.pred_str_gen(batch)
        """This returns the count of unique entities in both subject and object positions."""
        return f"""SELECT ?p1 (COUNT(DISTINCT ?e) AS ?entities) WHERE {{
            VALUES (?p1) {{ {pred_str}}}
            {{?e ?p1 ?o .
            FILTER(isURI(?e))}}
            UNION {{?s ?p1 ?e .
            FILTER(isURI(?e))}}
        }}
        GROUP BY ?p1
        """

    def unique_obj_for_predicate(batch):
        """This returns the count of unique object entities."""
        pred_str = PredicateStatQueries.pred_str_gen(batch)
        return f"""SELECT ?p1 (COUNT(DISTINCT ?e) AS ?entities) WHERE {{
            VALUES (?p1) {{ {pred_str}}}
            ?s ?p1 ?e .
            FILTER(isURI(?e))
        }}
        GROUP BY ?p1
        """

    def unique_subj_for_predicate(batch):
        """This returns the count of unique object entities."""
        pred_str = PredicateStatQueries.pred_str_gen(batch)
        return f"""SELECT ?p1 (COUNT(DISTINCT ?e) AS ?entities) WHERE {{
            VALUES (?p1) {{ {pred_str}}}
            ?e ?p1 ?o . 
        }}
        GROUP BY ?p1
        """

    def frequency_for_predicate(batch):
        pred_str = PredicateStatQueries.pred_str_gen(batch)
        return f"""SELECT ?p1 (COUNT(*) AS ?triples) WHERE {{
            VALUES (?p1) {{ {pred_str}}}
            ?s ?p1 ?o .
        }}
        GROUP BY ?p1
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Predicate Stat Extractor",
        description="Extract Predicate stats",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("task")
    parser.add_argument("-e", "--endpoint")
    parser.add_argument("--dir", "--output_dir")
    parser.add_argument("--input_dir")
    parser.add_argument("--pred_file")
    parser.add_argument("--batch_start")
    parser.add_argument("--batch_end")

    args = parser.parse_args()

    if args.task == "extract-predicates-stat":
        output_dir = f"{args.dir}/predicate"
        os.system(f"mkdir -p {output_dir}")
        endpoint = Endpoint(args.endpoint)
        # input_dir = f"{args.input_dir}/predicate"
        input_dir = f"{args.input_dir}"
        os.system(f"mkdir -p {args.dir}")
        os.system(f"mkdir -p {input_dir}")
        extrator = PredicateFreqExtractor(
            endpoint, input_dir, args.dir, predicate_file=args.pred_file
        )
        # extrator.query_batches()
        extrator.query_batches(int(args.batch_start), int(args.batch_end))
    elif args.task == "extract-predicates-stat-sub-obj":
        output_dir = f"{args.dir}/predicate"
        os.system(f"mkdir -p {output_dir}")
        endpoint = Endpoint(args.endpoint)
        # input_dir = f"{args.input_dir}/predicate"
        input_dir = f"{args.input_dir}"
        os.system(f"mkdir -p {args.dir}")
        os.system(f"mkdir -p {input_dir}")
        extrator = PredicateFreqExtractor(
            endpoint, input_dir, args.dir, predicate_file=args.pred_file
        )
        # extrator.query_batches()
        extrator.query_batches_subj_obj(int(args.batch_start), int(args.batch_end))
