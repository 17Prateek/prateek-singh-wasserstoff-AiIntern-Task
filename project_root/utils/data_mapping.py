import json

def map_objects_to_data(objects, descriptions, extracted_texts):
    mapped_data = []
    
    for idx, obj in enumerate(objects):
        data = {
            "object_id": idx,
            "description": descriptions[idx],
            "extracted_text": extracted_texts[idx]
        }
        mapped_data.append(data)
    
    return mapped_data

def save_mapping_to_json(mapped_data, output_file):
    with open(output_file, "w") as f:
        json.dump(mapped_data, f)
