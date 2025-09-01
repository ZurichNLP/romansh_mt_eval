from datasets import load_dataset
import pandas as pd
import numpy as np

class DatasetParser:
    def __init__(self, translation_dataset, img_dataset):
        self.translation_dataset = translation_dataset
        self.img_dataset = img_dataset
        self.translation_ds_config_name = "en-de_DE"

    def load_datasets(self):
        ds = load_dataset(self.translation_dataset, self.translation_ds_config_name)
        images = load_dataset(self.img_dataset)

        filtered_ds = ds.filter(lambda x: x["is_bad_source"] is False)

        df = pd.DataFrame(filtered_ds["train"])
        df_images = pd.DataFrame(images["test"])

        df_images["url"] = df_images["original_url"].combine_first(df_images["mirror_url"])
        merged_df = df.merge(df_images[["document_id", "url"]], on="document_id", how="left")
        is_duplicate_doc_id = merged_df['document_id'].duplicated(keep='first')
        merged_df.loc[is_duplicate_doc_id, 'url'] = ""
        return merged_df
