import torch, configargparse
from data import load_asap_data
from model_architechure_bert_multi_scale_multi_loss import DocumentBertScoringModel


def _initialize_arguments(p: configargparse.ArgParser):
    p.add('--bert_model_path', help='bert_model_path')
    p.add('--efl_encode', action='store_true', help='is continue training')
    p.add('--r_dropout', help='r_dropout', type=float)
    p.add('--batch_size', help='batch_size', type=int)
    p.add('--bert_batch_size', help='bert_batch_size', type=int)
    p.add('--cuda', action='store_true', help='use gpu or not')
    p.add('--device')
    p.add('--model_directory', help='model_directory')
    p.add('--test_file', help='test data file')
    p.add('--data_dir', help='data directory to store asap experiment data')
    p.add('--data_sample_rate', help='data_sample_rate', type=float)
    p.add('--prompt', help='prompt')
    p.add('--fold', help='fold')
    p.add('--chunk_sizes', help='chunk_sizes', type=str)
    p.add('--result_file', help='pred result file path', type=str)

    args = p.parse_args()
    args.test_file = "%s/p8_fold3_test.txt" % args.data_dir
    args.model_directory = "%s/%s_%s" % (args.model_directory, args.prompt, args.fold)
    args.bert_model_path = args.model_directory

    if torch.cuda.is_available() and args.cuda:
        args.device = 'cuda'
    else:
        args.dev = 'cpu'
    return args


if __name__ == "__main__":
    # initialize arguments
    p = configargparse.ArgParser(default_config_files=["asap.ini"])
    args = _initialize_arguments(p)
    print(args)

    # load data
    test = load_asap_data(args.test_file)

    test_documents, test_labels = [], []
    for _, text, label in test:
        test_documents.append(text)
        test_labels.append(label)

    print("sample number:", len(test_documents))
    print("label number:", len(test_labels))

    model = DocumentBertScoringModel(args=args)
    model.predict_for_regress((test_documents, test_labels))
