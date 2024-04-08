
from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Union, tuple

import pandas as pd
import polars as pl
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from utils.utils import load_dataset, load_dataset_list


class Preprocessor:
    """Preprocesses numerical, ordinal and unstructured text.

    Static preprocessing: Cleans, trims, aggregates and saves text into text blocks
    During runtime it encodes ordinal and categorical features,
    normalizes numerical features and returns

    Attributes
    ----------
        text_columns: All columns supposed to be treated as text
        textblocks: A dict of all columns associated with each text block for aggregation
        ordinal_columns: All columns supposed to be Ordinal encoded
        ordinal_categories: All categories in ascending order for each ordinal column
        one_hot_columns: All columns supposed to be One Hot encoded
        numerical_columns: All columns supposed to be normalized
        trim_dict: A dict containing all text columns and their desired char length
        separation_token: Separation token used during text block aggregation
        stopwords: A set of all stopwords to be removed during static preprocessing
        save_path: Path to save dataframes to after static preprocessing

    """

    def __init__(
        self,
        numerical_columns: list[str],
        ordinal_columns: list[str],
        ordinal_categories: list[str],
        one_hot_columns: list[str],
        separation_token: str = '[SEP]',
        text_columns: Union[list, None] = None,
        textblocks: Union[list[list[str]], None] = None,
        trim_dict: Union[dict[str, int], None] = None,
        save_path: Union[Path, None] = None,
    ) -> None:
        self.text_columns = text_columns
        self.textblocks = textblocks
        self.ordinal_columns = list(ordinal_columns)
        self.one_hot_columns = list(one_hot_columns)
        self.numerical_columns = list(numerical_columns)
        self.trim_dict = trim_dict
        self.separation_token = separation_token
        self.stopwords = setup_stopwords()
        self.save_path = save_path
        self.ordinal_categories = ordinal_categories
        self.scaler = MinMaxScaler()

    def _preprocess_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans text columns in dataframe.

        Lowercase's all text, removes stopwords, expands german contractions
        removes urls and removes special characters.

        Args:
        ----
            df: DataFrame containing all text columns.

        Returns: DataFrame with sanitized text columns.

        """
        df[['3_skills', '4_interests']] = df[['3_skills', '4_interests']].replace(
            to_replace=',',
            value=' ',
        )
        df[self.text_columns] = (
            df[self.text_columns]
            .apply(lambda x: x.str.lower())
            .replace(
                dict.fromkeys(self.stopwords, ''),
                inplace=False,
            )
            .replace({'ä': 'ae', 'ü': 'ue', 'ö': 'oe'})
            .replace(r'http\S+', '', regex=True)
            .replace(r'[^A-Za-z0-9 ]+', '', regex=True)
        )
        return df

    def static_preprocessing(
        self,
        dataset_folder: str,
        input_format: str = 'csv',
        output_format: str = 'parquet',
    ) -> None:
        """Entry point for static preprocessing

        Cleans and aggregates text columns of a dataframes.
        Optionally saves treated Dataframes.

        Args:
        ----
            dataset_folder: Path object to folder containing dataframes.
            input_format: format of input dataframes.
            output_format: format of saved dataframes if save path exist in preprocessor.

        """
        datasets = load_dataset_list(
            path=dataset_folder,
            file_pattern=input_format,
        )
        for idx, dataset in enumerate(datasets):
            print(f'Processing {idx + 1}, file {dataset}')
            df = load_dataset(
                dataset=dataset,
                backend='pandas',
            )
            if self.trim_dict:
                df = self.trim_text(
                    df=df,
                )
            df = self._preprocess_text(
                df=df,
            )
            text_block_df = self.construct_textblocks(
                df=df,
            )
            to_drop = list(chain.from_iterable(self.textblocks))
            df = df.drop(
                to_drop,
                axis=1,
            )
            df = pd.concat(
                [text_block_df, df],
                axis=1,
            )
            if self.save_path:
                if output_format == 'parquet':
                    df.to_parquet(
                        str(self.save_path) + f'/processed_dataset_{idx}.parquet',
                        index=False,
                    )
                elif output_format == 'csv':
                    df.to_csv(
                        str(self.save_path) + f'/processed_dataset_{idx}.csv',
                        index=False,
                    )

    def preprocess_numerical(
        self,
        dataset: pd.DataFrame,
        dataset_type: str,
    ) -> pl.DataFrame:
        """Applies Min-Max scaling to numerical columns fitted on trainings data.

        Args:
        ----
            dataset: Dataframe containing numerical features with column names from self.numerical_columns.
            dataset_type: If set to 'train' triggers a fitting of scaler.

        Returns: Scaled numerical features.

        """
        if dataset_type == 'train':
            self.scaler = self.scaler.fit(
                dataset[self.numerical_columns],
            )
        scaled_num_features = self.scaler.transform(dataset[self.numerical_columns])
        return scaled_num_features

    def construct_encodings(
        self,
        df: pd.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Constructs One-hot and non normalized ordinal encodings.

        Args:
        ----
            df: Dataframe containing all One-hot and ordinal columns.

        Returns: Dataframes containing One-hot and ordinal encodings.

        """
        onh_enc = OneHotEncoder(handle_unknown='error', sparse_output=False)
        onh_enc_df = onh_enc.fit_transform(
            df[self.one_hot_columns],
        )
        ord_enc = OrdinalEncoder(
            handle_unknown='error',
            categories=[self.ordinal_categories],
        )
        ord_enc_df = ord_enc.fit_transform(df[self.ordinal_columns])
        return (onh_enc_df, ord_enc_df)

    def construct_textblocks(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Constructs text blocks by aggregating columns concatenated with separation token.

        Args:
        ----
            df:Dataframe containing all columns in text blocks.

        Returns: Dataframe containing all text blocks.

        """
        textblock_df = pd.DataFrame()
        for idx, col_set in enumerate(self.textblocks):
            textblock_df['textblock_' + str(idx)] = (
                df[col_set]
                .apply(
                    lambda x: self.separation_token.join(x.astype(str).values),
                    axis=1,
                )
                .replace({'nan': ''}, regex=True)
            )
        return textblock_df

    def trim_text(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame():
        for col, length in self.trim_dict.items():
            df[col] = df[col].astype(str).str.slice(0, length)
        return df


def setup_stopwords() -> set:
    list_of_stopwords = set()
    for language in stopwords.fileids():
        list_of_stopwords = list_of_stopwords | set(stopwords.words(language))
    return list_of_stopwords


if __name__ == '__main__':
    path = str(Path().absolute() / 'processed_data')
    numerical_columns = ['0_linkedin_connections', '0_job_start_date']
    one_hot_columns = ['0_industry', '0_job_company_industry']
    ordinal_columns = ['0_inferred_salary']
    text_columns = [
        '0_gender',
        '0_industry',
        '0_job_company_industry',
        '0_job_company_location_country',
        '0_job_company_location_region',
        '0_job_company_name',
        '0_job_title',
        '0_location_country',
        '0_location_region',
        '0_summary',
        '3_skills',
        '4_interests',
        '5_certifications',
        '6_education_degree_0',
        '6_education_degree_1',
        '6_education_degree_2',
        '6_education_degree_3',
        '6_education_degree_major_0',
        '6_education_degree_major_1',
        '6_education_degree_major_2',
        '6_education_degree_major_3',
        '6_education_degree_minor_0',
        '6_education_degree_minor_1',
        '6_education_degree_minor_2',
        '6_education_degree_minor_3',
        '6_school_country_0',
        '6_school_country_1',
        '6_school_country_2',
        '6_school_country_3',
        '6_school_name_0',
        '6_school_name_1',
        '6_school_name_2',
        '6_school_name_3',
        '6_school_region_0',
        '6_school_region_1',
        '6_school_region_2',
        '6_school_region_3',
        '7_company_country_0',
        '7_company_country_1',
        '7_company_country_2',
        '7_company_country_3',
        '7_company_country_4',
        '7_company_country_5',
        '7_company_country_6',
        '7_company_country_7',
        '7_company_country_8',
        '7_company_industry_0',
        '7_company_industry_1',
        '7_company_industry_2',
        '7_company_industry_3',
        '7_company_industry_4',
        '7_company_industry_5',
        '7_company_industry_6',
        '7_company_industry_7',
        '7_company_industry_8',
        '7_company_name_0',
        '7_company_name_1',
        '7_company_name_2',
        '7_company_name_3',
        '7_company_name_4',
        '7_company_name_5',
        '7_company_name_6',
        '7_company_name_7',
        '7_company_name_8',
        '7_company_region_0',
        '7_company_region_1',
        '7_company_region_2',
        '7_company_region_3',
        '7_company_region_4',
        '7_company_region_5',
        '7_company_region_6',
        '7_company_region_7',
        '7_company_region_8',
        '7_job_title_0',
        '7_job_title_1',
        '7_job_title_2',
        '7_job_title_3',
        '7_job_title_4',
        '7_job_title_5',
        '7_job_title_6',
        '7_job_title_7',
        '7_job_title_8',
        '7_location_name_0',
        '7_location_name_1',
        '7_location_name_2',
        '7_location_name_3',
        '7_location_name_4',
        '7_location_name_5',
        '7_location_name_6',
        '7_location_name_7',
        '7_location_name_8',
    ]
    textblocks = [
        [
            '0_gender',
            '0_job_company_founded',
            '0_job_company_location_country',
            '0_job_company_location_region',
            '0_job_company_name',
            '0_job_company_size',
            '0_job_title',
            '0_job_title_role',
            '0_location_country',
            '0_location_region',
            '0_summary',
            '2_language_name_0',
            '2_language_name_1',
            '2_language_name_2',
        ],
        ['3_skills', '4_interests', '5_certifications'],
        [
            '6_education_degree_0',
            '6_education_degree_1',
            '6_education_degree_2',
            '6_education_degree_3',
            '6_education_degree_major_0',
            '6_education_degree_major_1',
            '6_education_degree_major_2',
            '6_education_degree_major_3',
            '6_education_degree_minor_0',
            '6_education_degree_minor_1',
            '6_education_degree_minor_2',
            '6_education_degree_minor_3',
            '6_education_end_0',
            '6_education_end_1',
            '6_education_end_2',
            '6_education_end_3',
        ],
        [
            '6_education_start_0',
            '6_education_start_1',
            '6_education_start_2',
            '6_education_start_3',
            '6_school_country_0',
            '6_school_country_1',
            '6_school_country_2',
            '6_school_country_3',
            '6_school_name_0',
            '6_school_name_1',
            '6_school_name_2',
            '6_school_name_3',
            '6_school_region_0',
            '6_school_region_1',
            '6_school_region_2',
            '6_school_region_3',
            '6_school_type_0',
            '6_school_type_1',
            '6_school_type_2',
            '6_school_type_3',
        ],
        [
            '7_company_country_0',
            '7_company_country_1',
            '7_company_country_2',
            '7_company_country_3',
            '7_company_country_4',
            '7_company_country_5',
            '7_company_country_6',
            '7_company_country_7',
            '7_company_country_8',
            '7_company_industry_0',
            '7_company_size_0',
            '7_company_size_1',
            '7_company_size_2',
            '7_company_size_3',
            '7_company_size_4',
            '7_company_size_5',
            '7_company_size_6',
            '7_company_size_7',
            '7_company_size_8',
            '7_job_end_0',
            '7_job_end_1',
            '7_job_end_2',
            '7_job_end_3',
            '7_job_end_4',
            '7_job_end_5',
            '7_job_end_6',
            '7_job_end_7',
            '7_job_end_8',
        ],
        [
            '7_company_industry_0',
            '7_company_industry_1',
            '7_company_industry_2',
            '7_company_industry_3',
            '7_company_industry_4',
            '7_company_industry_5',
            '7_company_industry_6',
            '7_company_industry_7',
            '7_company_industry_8',
            '7_job_level_0',
            '7_job_level_1',
            '7_job_level_2',
            '7_job_level_3',
            '7_job_level_4',
            '7_job_level_5',
            '7_job_level_6',
            '7_job_level_7',
            '7_job_level_8',
        ],
        [
            '7_company_name_0',
            '7_company_name_1',
            '7_company_name_2',
            '7_company_name_3',
            '7_company_name_4',
            '7_company_name_5',
            '7_company_name_6',
            '7_company_name_7',
            '7_company_name_8',
            '7_job_start_0',
            '7_job_start_1',
            '7_job_start_2',
            '7_job_start_3',
            '7_job_start_4',
            '7_job_start_5',
            '7_job_start_6',
            '7_job_start_7',
            '7_job_start_8',
        ],
        [
            '7_company_region_0',
            '7_company_region_1',
            '7_company_region_2',
            '7_company_region_3',
            '7_company_region_4',
            '7_company_region_5',
            '7_company_region_6',
            '7_company_region_7',
            '7_company_region_8',
        ],
        [
            '7_job_title_0',
            '7_job_title_1',
            '7_job_title_2',
            '7_job_title_3',
            '7_job_title_4',
            '7_job_title_5',
            '7_job_title_6',
            '7_job_title_7',
            '7_job_title_8',
        ],
        [
            '7_location_name_0',
            '7_location_name_1',
            '7_location_name_2',
            '7_location_name_3',
            '7_location_name_4',
            '7_location_name_5',
            '7_location_name_6',
            '7_location_name_7',
            '7_location_name_8',
        ],
    ]
    trim_dict = {
        '0_gender': 6,
        '0_job_company_founded': 6,
        '0_job_company_location_country': 20,
        '0_job_company_location_region': 30,
        '0_job_company_name': 30,
        '0_job_company_size': 12,
        '0_job_title': 30,
        '0_job_title_role': 16,
        '0_location_country': 20,
        '0_location_region': 30,
        '0_summary': 199,
        '2_language_name_0': 16,
        '2_language_name_1': 16,
        '2_language_name_2': 16,
        '3_skills': 262,
        '4_interests': 120,
        '5_certifications': 120,
        '6_education_degree_0': 39,
        '6_education_degree_1': 39,
        '6_education_degree_2': 39,
        '6_education_degree_3': 39,
        '6_education_degree_major_0': 30,
        '6_education_degree_major_1': 30,
        '6_education_degree_major_2': 30,
        '6_education_degree_major_3': 30,
        '6_education_degree_minor_0': 30,
        '6_education_degree_minor_1': 30,
        '6_education_degree_minor_2': 30,
        '6_education_degree_minor_3': 30,
        '6_education_end_0': 6,
        '6_education_end_1': 6,
        '6_education_end_2': 6,
        '6_education_end_3': 6,
        '6_education_start_0': 6,
        '6_education_start_1': 6,
        '6_education_start_2': 6,
        '6_education_start_3': 6,
        '6_school_country_0': 20,
        '6_school_country_1': 20,
        '6_school_country_2': 20,
        '6_school_country_3': 20,
        '6_school_name_0': 36,
        '6_school_name_1': 36,
        '6_school_name_2': 36,
        '6_school_name_3': 36,
        '6_school_region_0': 25,
        '6_school_region_1': 25,
        '6_school_region_2': 25,
        '6_school_region_3': 25,
        '6_school_type_0': 16,
        '6_school_type_1': 16,
        '6_school_type_2': 16,
        '6_school_type_3': 16,
        '7_company_country_0': 25,
        '7_company_country_1': 25,
        '7_company_country_2': 25,
        '7_company_country_3': 25,
        '7_company_country_4': 25,
        '7_company_country_5': 25,
        '7_company_country_6': 25,
        '7_company_country_7': 25,
        '7_company_country_8': 25,
        '7_company_industry_0': 38,
        '7_company_industry_1': 38,
        '7_company_industry_2': 38,
        '7_company_industry_3': 38,
        '7_company_industry_4': 38,
        '7_company_industry_5': 38,
        '7_company_industry_6': 38,
        '7_company_industry_7': 38,
        '7_company_industry_8': 38,
        '7_company_name_0': 40,
        '7_company_name_1': 40,
        '7_company_name_2': 40,
        '7_company_name_3': 40,
        '7_company_name_4': 40,
        '7_company_name_5': 40,
        '7_company_name_6': 40,
        '7_company_name_7': 40,
        '7_company_name_8': 40,
        '7_company_region_0': 51,
        '7_company_region_1': 51,
        '7_company_region_2': 51,
        '7_company_region_3': 51,
        '7_company_region_4': 51,
        '7_company_region_5': 51,
        '7_company_region_6': 51,
        '7_company_region_7': 51,
        '7_company_region_8': 51,
        '7_company_size_0': 10,
        '7_company_size_1': 10,
        '7_company_size_2': 10,
        '7_company_size_3': 10,
        '7_company_size_4': 10,
        '7_company_size_5': 10,
        '7_company_size_6': 10,
        '7_company_size_7': 10,
        '7_company_size_8': 10,
        '7_job_end_0': 6,
        '7_job_end_1': 6,
        '7_job_end_2': 6,
        '7_job_end_3': 6,
        '7_job_end_4': 6,
        '7_job_end_5': 6,
        '7_job_end_6': 6,
        '7_job_end_7': 6,
        '7_job_end_8': 6,
        '7_job_level_0': 8,
        '7_job_level_1': 8,
        '7_job_level_2': 8,
        '7_job_level_3': 8,
        '7_job_level_4': 8,
        '7_job_level_5': 8,
        '7_job_level_6': 8,
        '7_job_level_7': 8,
        '7_job_level_8': 8,
        '7_job_start_0': 6,
        '7_job_start_1': 6,
        '7_job_start_2': 6,
        '7_job_start_3': 6,
        '7_job_start_4': 6,
        '7_job_start_5': 6,
        '7_job_start_6': 6,
        '7_job_start_7': 6,
        '7_job_start_8': 6,
        '7_job_title_0': 51,
        '7_job_title_1': 51,
        '7_job_title_2': 51,
        '7_job_title_3': 51,
        '7_job_title_4': 51,
        '7_job_title_5': 51,
        '7_job_title_6': 51,
        '7_job_title_7': 51,
        '7_job_title_8': 51,
        '7_location_name_0': 51,
        '7_location_name_1': 51,
        '7_location_name_2': 51,
        '7_location_name_3': 51,
        '7_location_name_4': 51,
        '7_location_name_5': 51,
        '7_location_name_6': 51,
        '7_location_name_7': 51,
        '7_location_name_8': 51,
    }

    ordinal_categories = [
        '<20,000',
        '20,000-25,000',
        '25,000-35,000',
        '35,000-45,000',
        '45,000-55,000',
        '55,000-70,000',
        '70,000-85,000',
        '85,000-100,000',
        '100,000-150,000',
        '150,000-250,000',
        '>250,000',
    ]

    save_path = Path().absolute() / 'encoded_data'
    preprocessor = Preprocessor(
        text_columns=text_columns,
        textblocks=textblocks,
        numerical_columns=numerical_columns,
        ordinal_columns=ordinal_columns,
        one_hot_columns=one_hot_columns,
        save_path=save_path,
        trim_dict=trim_dict,
        ordinal_categories=ordinal_categories,
    )
    preprocessor.static_preprocessing(path)
