import json

if __name__ == "__main__":
    base_dir = "../docs/pii_data"
    groundtruth_files = [
        f"{base_dir}/nytimes/ground_truth.json",
        f"{base_dir}/openai/ground_truth.json",
        f"{base_dir}/techcrunch/ground_truth.json",
    ]
    # Check all the groundtruth files
    for file_dir in groundtruth_files:
        with open(file_dir, "r") as f:
            data = json.load(f)
        #
        for url, entities in data.items():
            # Get js file name
            src_webpage = file_dir.split("/")[3]
            js_file = url.split("/")[-1].split("?")[0]

            # Read the file
            js_file_dir = f"{base_dir}/{src_webpage}/raw_data/{js_file}"

            try:
                # Read the script
                f = open(js_file_dir, "r")
                lines = f.readlines()

                for line_num, line in enumerate(lines):
                    for entity in entities:
                        if entity["line_number"] == line_num:
                            start = entity["start_position"]
                            end = entity["end_position"]
                            print(line[start:end])

            except:
                pass

            # for entity in entities:

            #     js_file_content[]
