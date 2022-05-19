from LSTMmodel import *
from utils import *
from init import *
from dataset import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--confidence",
        help="Confidence score to predict",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "-ep", "--epochs", help="Number of training epochs", type=int, default=50
    )
    parser.add_argument(
        "-in_seq",
        "--input_seq_len",
        help="input sequence length",
        type=int,
        default=180,
    )
    parser.add_argument(
        "-out_seq",
        "--output_seq_len",
        help="output sequence length",
        type=int,
        default=30,
    )
    parser.add_argument(
        "-lr", "--learning_rate", help="learning rate", type=float, default=0.00001
    )
    parser.add_argument(
        "-coef", "--coefficient_loss", help="coefficent loss", type=float, default=0.005
    )
    parser.add_argument(
        "-path",
        "--model_path",
        help="path of save model",
        type=str,
        default="checkpoint/smallLSTM.pth",
    )
    parser.add_argument(
        "-syn_thresh",
        "--synthetic_threshold",
        help="synthetic_threshold",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "-syn_seq",
        "--synthetic_seq_len",
        help="synthetic sequence length",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    input_paths = "data/test"
    list_path = os.listdir(input_paths)
    # list_path = ["data/test/fimi_6_01-05-2022_05-05-2022.csv"]
    for index, input_path in enumerate(list_path):
        if "fimi_6" in input_path or "fimi_18" in input_path or "fimi_30" in input_path:
            print("input_path: ", input_path)
            test_df, test_iterator, sc_test = prepare_test(
                input_path=os.path.join(input_paths, input_path),
                synthetic_threshold=args.synthetic_threshold,
                synthetic_sequence_length=args.synthetic_seq_len,
                input_len=args.input_seq_len,
                output_len=args.output_seq_len,
            )

            model = LSTM(
                input_seq_len=args.input_seq_len,
                output_seq_len=args.output_seq_len,
                confidence=args.confidence,
                number_layer=1,
                input_size=2,
                hidden_size=64,
            )
            model.to(device)
            # print(model)
            # test phase
            model.load_state_dict(copyStateDict(torch.load(args.model_path)))
            # print('test batch')
            print(args.model_path)
            # model.predict(
            #     iterator=test_iterator, sc_test=sc_test, confidence=args.confidence
            # )

            # inference
            import time

            # print('inference...')
            list_confidence = [
                # 0.2,
                # 0.25,
                0.3,
                # 0.35,
                # 0.4,
                # 0.45,
                0.5,
                # 0.55,
                # 0.6,
                # 0.65,
                # 0.7,
                # 0.75,
                0.8,
                # 0.85,
                # 0.9,
            ]
            # s = []
            # mape = []

            # for c in list_confidence:
            #     st = time.time()
            #     print('confidence: ', c)
            #     out1, out2 = model.eval_realtime_2(test_df=test_df,
            #                         input_length=args.input_seq_len,
            #                         output_length=args.output_seq_len,
            #                         confidence=c,
            #                         sc_test=sc_test,
            #                         synthetic_threshold=args.synthetic_threshold,
            #                         synthetic_seq_len=args.synthetic_seq_len)
            #     print('take: ', time.time() - st)
            #     s.append(out1)
            #     mape.append(out2)

            # dicts = {'saving': s, 'mape': mape}

            # df_result = pd.DataFrame(dicts)
            # df_result.to_csv(f'result/smallLSTM_inference2_data_{index}.csv')

            s = []
            mape = []

            for c in list_confidence:
                st = time.time()
                print("confidence: ", c)
                out1, out2 = model.eval_realtime_2(
                    test_df=test_df,
                    input_length=args.input_seq_len,
                    output_length=args.output_seq_len,
                    confidence=c,
                    sc_test=sc_test,
                    synthetic_threshold=args.synthetic_threshold,
                    synthetic_seq_len=args.synthetic_seq_len,
                    name=input_path[:-4],
                )
                print("take: ", time.time() - st)
                s.append(out1)
                mape.append(out2)

            dicts = {"saving": s, "mape": mape}

            df_result = pd.DataFrame(dicts)
            df_result.to_csv(f"result/test/smallLSTM_inference_{index}.csv")
