#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info("Downloading input artifact")
    artifact_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading dataset")
    df = pd.read_csv(artifact_path)

    logger.info("Handling missing values")
    if 'reviews_per_month' in df.columns:
        df['reviews_per_month'].fillna(0, inplace=True)

    logger.info("Filtering price outliers")
    min_price = args.min_price
    max_price = args.max_price
    if 'price' in df.columns:
        idx = df['price'].between(min_price, max_price)
        df = df[idx].copy()

    logger.info("Filtering geographic boundaries")
    if 'longitude' in df.columns and 'latitude' in df.columns:
        idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
        df = df[idx].copy()

    if 'last_review' in df.columns:
        logger.info("Converting last_review to datetime")
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

    output_filename = "clean_sample.csv"
    logger.info(f"Saving cleaned data to {output_filename}")
    df.to_csv(output_filename, index=False)

    logger.info("Creating output artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_filename)

    logger.info("Logging artifact to W&B")
    run.log_artifact(artifact)

    logger.info("Removing temporary file")
    os.remove(output_filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very vasic data cleaning")
    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact to download from W&B",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact to upload to W&B",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=int,
        help="Minimum price to filter the dataset",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=int,
        help="Maximum price to filter the dataset",
        required=True
    )


    args = parser.parse_args()

    go(args)
