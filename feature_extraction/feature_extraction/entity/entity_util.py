from feature_extraction.predicates.pred_util import *


class EntityFreqExtractor(ExtractorBase):
    def __init__(
        self,
        endpoint: Endpoint,
        input_dir: str,
        output_dir: str,
        entity_file="entity.json",
    ) -> None:
        super().__init__(endpoint, output_dir, entity_file)
        self.input_dir = input_dir
        self.entity_file = entity_file
        self.entities = json.load(open(entity_file, "r"))
        # for backward compatibility
        self.predicates = self.entities
        self.batch_output_dir = f"{output_dir}/ent_stat/batches"
        os.system(f"mkdir -p {self.batch_output_dir}")
        # the
        # path to where the responses are saved with the features.
        self.batch_output_response_dir = f"{output_dir}/ent_stat/batches_response_stats"
        os.system(f"mkdir -p {self.batch_output_response_dir}")
        os.system(f"mkdir -p {self.batch_output_response_dir}/freq")
        os.system(f"mkdir -p {self.batch_output_response_dir}/subj")
        os.system(f"mkdir -p {self.batch_output_response_dir}/obj")
        self.load_batches()

    def query_batches(self, batch_start=1, batch_end=2):
        if not hasattr(self, "batches"):
            self.load_batches()
        save_path = self.batch_output_response_dir
        os.system(f"mkdir -p {save_path}")
        print(f"Entity Stats are saved to: {save_path}")
        for i, b in enumerate(self.batches[batch_start - 1 : batch_end - 1]):
            for query_generator, name in zip(
                [
                    EntityStatQueries.freq_entity,
                    EntityStatQueries.sub_entity,
                    EntityStatQueries.obj_entity,
                ],
                ["freq", "subj", "obj"],
            ):
                query = query_generator(b)
                res = self.endpoint.run_query(query)
                res_fp = f"{save_path}/{name}/batch_{batch_start+i}.json"
                json.dump(res, open(res_fp, "w"))
                print(f"batch {batch_start+i} extracted!")


class EntityStatQueries:
    def pred_str_gen(batch):
        pred_str = ""
        for p in batch:
            if " " in p:
                raise Exception("space in pred")
            pred_str += f"(<{p}>) "
        return pred_str

    def freq_entity(batch):
        ent_str = EntityStatQueries.pred_str_gen(batch)
        """This returns the count of unique entities in both subject and object positions."""
        return f"""SELECT ?e (COUNT(DISTINCT *) AS ?entities) WHERE {{
            VALUES (?e) {{ {ent_str}}}
            {{?e ?p1 ?o .}}
            UNION {{?s ?p2 ?e .}}
        }}
        GROUP BY ?e
        """

    def sub_entity(batch):
        ent_str = EntityStatQueries.pred_str_gen(batch)
        """This returns the count of unique entities in both subject and object positions."""
        return f"""SELECT ?e (COUNT(DISTINCT *) AS ?entities) WHERE {{
            VALUES (?e) {{ {ent_str}}}
            {{?e ?p1 ?o .}}
        }}
        GROUP BY ?e
        """

    def obj_entity(batch):
        ent_str = EntityStatQueries.pred_str_gen(batch)
        """This returns the count of unique entities in both subject and object positions."""
        return f"""SELECT ?e (COUNT(DISTINCT *) AS ?entities) WHERE {{
            VALUES (?e) {{ {ent_str}}}
            {{?s ?p2 ?e .}}
        }}
        GROUP BY ?e
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Entity Stat Extractor",
        description="Extract entity stats",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("task")
    parser.add_argument("-e", "--endpoint")
    parser.add_argument("--dir", "--output_dir")
    parser.add_argument("--input_dir")
    parser.add_argument("--ent_file")
    parser.add_argument("--batch_start")
    parser.add_argument("--batch_end")

    args = parser.parse_args()

    if args.task == "extract-entity-stat":
        output_dir = f"{args.dir}"
        os.system(f"mkdir -p {output_dir}")
        endpoint = Endpoint(args.endpoint)
        # input_dir = f"{args.input_dir}/predicate"
        input_dir = f"{args.input_dir}"
        os.system(f"mkdir -p {args.dir}")
        os.system(f"mkdir -p {input_dir}")
        extrator = EntityFreqExtractor(
            endpoint, input_dir, args.dir, entity_file=args.ent_file
        )
        # extrator.query_batches()
        extrator.query_batches(int(args.batch_start), int(args.batch_end))