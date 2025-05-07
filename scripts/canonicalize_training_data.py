import sqlite3
import pandas as pd
import argparse
from typing import Set, List, Dict

class SimpleNodeSynonymizer:
    def __init__(self, sqlite_file_path: str):
        self.database_path = sqlite_file_path
        if not sqlite_file_path or not sqlite3.connect(sqlite_file_path):
            raise ValueError(f"Specified synonymizer database does not exist or is inaccessible: {sqlite_file_path}")
        self.db_connection = sqlite3.connect(sqlite_file_path)

    def __del__(self):
        if hasattr(self, "db_connection"):
            self.db_connection.close()

    def get_canonical_curies_in_batches(self, curies: Set[str], batch_size: int = 5000) -> Dict[str, Dict]:
        results_dict = {}
        curie_batches = self._batch_curies(curies, batch_size)

        for batch in curie_batches:
            sql_query_template = """
                SELECT N.id_simplified, N.cluster_id, C.name, C.category
                FROM nodes AS N
                INNER JOIN clusters AS C ON C.cluster_id == N.cluster_id
                WHERE N.id_simplified IN ({});
            """
            placeholders = ', '.join(f"'{curie}'" for curie in batch)
            sql_query = sql_query_template.format(placeholders)
            cursor = self.db_connection.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            cursor.close()

            for row in rows:
                curie, canonical_curie, canonical_name, canonical_category = row
                results_dict[curie] = {
                    "preferred_curie": canonical_curie,
                    "preferred_name": canonical_name,
                    "preferred_category": f"biolink:{canonical_category}" if canonical_category else None
                }

        for curie in curies:
            if curie not in results_dict:
                results_dict[curie] = None

        return results_dict

    @staticmethod
    def _batch_curies(curies: Set[str], batch_size: int) -> List[Set[str]]:
        curie_list = list(curies)
        return [set(curie_list[i:i + batch_size]) for i in range(0, len(curie_list), batch_size)]

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Canonicalize node IDs in a TSV file using a synonymizer database.")
    parser.add_argument("input_file", help="Path to the input TSV file")
    parser.add_argument("sqlite_file_path", help="Path to the node synonymizer SQLite file")
    parser.add_argument("output_file", help="Path to save the canonicalized output TSV file")
    args = parser.parse_args()

    input_file = args.input_file
    sqlite_file_path = args.sqlite_file_path
    output_file = args.output_file

    df = pd.read_csv(input_file, sep="\t")
    
    if "node1_id" not in df.columns:
        raise ValueError("Input file does not contain the 'node1_id' column.")
    if "node2_id" not in df.columns:
        raise ValueError("Input file does not contain the 'node2_id' column.")
    
    curies = set(df["node1_id"].dropna().str.upper().unique())
    curies.update(df["node2_id"].dropna().str.upper().unique())

    # Synonymizer
    synonymizer = SimpleNodeSynonymizer(sqlite_file_path)
    results = synonymizer.get_canonical_curies_in_batches(curies, batch_size=10000)

    def get_preferred_curie(curie):
        if pd.isna(curie) or curie == "":
            return None
        curie_info = results.get(curie.upper())
        if curie_info is None:
            return None
        return curie_info.get("preferred_curie", None)

    # Canonicalize
    df["node1_id"] = df["node1_id"].apply(get_preferred_curie)
    df = df[df["node1_id"].notna()]
    df["node2_id"] = df["node2_id"].apply(get_preferred_curie)
    df = df[df["node2_id"].notna()]

    # Save
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved canonicalized training data to {output_file}")

if __name__ == "__main__":
    main()
