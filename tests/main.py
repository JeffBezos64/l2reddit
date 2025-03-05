def main():
    import torch
    import os
    import pickle
    try:
        from l2reddit.databalancer import DataBalancer
        from l2reddit.dataprocessor import DataProcessor
    except:
        print("not good enough")
        exit()
    # data_balancer = DataBalancer(random_seed=42)
    # data_balancer.get_statistics('/csse/research/NativeLanguageID/mthesis-phonological/experiment/data/text_chunks/original/europe_data')
    # data_balancer.get_statistics('/csse/research/NativeLanguageID/mthesis-phonological/experiment/data/text_chunks/original/non_europe_data')
    # data_balancer.create_balanced_folder(raw_data_dir='/csse/research/NativeLanguageID/mthesis-phonological/experiment/data/text_chunks/original/europe_data',
    #                                 output_dir='/csse/research/NativeLanguageID/mthesis-phonological/experiment/data/balanced/seed_42/europe_data',
    #                                 max_chunks_per_author=3,
    #                                 authors_per_language=104,
    #                                 )
    # data_balancer.create_balanced_folder(raw_data_dir='/csse/research/NativeLanguageID/mthesis-phonological/experiment/data/text_chunks/original/non_europe_data',
    #                                 output_dir='/csse/research/NativeLanguageID/mthesis-phonological/experiment/data/balanced/seed_42/non_europe_data',
    #                                 max_chunks_per_author=17,
    #                                 authors_per_language=273,
    #                                 )

    data_processor = DataProcessor('google/bigbird-roberta-base')
    data_processor.discover_chunks('/csse/research/NativeLanguageID/mthesis-phonological/experiment/data/balanced/seed_42/non_europe_data')

    #since 50% is used for finetune and testing and then the remaining half is used for the experiment.
    train_dataset, test_dataset = data_processor.get_train_test_datasets(split_by_chunks=True, seed=42, sequence_length=2048, train_size=0.5)
    with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/out_of_domain_experiment_dataframe_clean_chunks.pkl') as f:
        pickle.dump(test_dataset, f)


    #fine_tune_dataset, train_dataset = data_processor.split_dataset(train_dataset, train_size=0.5, shuffle=True, seed=seed)
    #fine_tune_dataset, fine_tune_validation_dataset = data_processor.split_dataset(fine_tune_dataset, train_size=0.9, shuffle=True, seed=seed)

    #train_dataloader = DataLoader(fine_tune_dataset, batch_size=batch_size, shuffle=False)
    #validation_dataloader = DataLoader(fine_tune_validation_dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    main()