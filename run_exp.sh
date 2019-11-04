nohup python -u main.py --cross_validate --use_pca --class_weight_scheme customize --model rf --additional_weight 0.2 --search_best_parameters --search_by_sklearn_api --random_search_n_iter 50

nohup python -u main.py --cross_validate --use_pca --search_best_parameters --search_by_sklearn_api --random_search_n_iter 50


for weight in 0.2 0.5 1.0 2.0 5.0 10.0
do

# python main.py --predict_mode --use_pca --force_retrain --event_wise --class_weight_scheme customize --model rf --additional_weight $weight
# python main.py --get_submission --use_pca --force_retrain --event_wise --class_weight_scheme customize --model rf --additional_weight $weight
echo "python main.py --get_submission --use_pca --force_retrain --event_wise --class_weight_scheme customize --model rf --additional_weight $weight"
python evaluate_2019B.py --input_file out/ensemble-customize/weight${weight}/submit-top-2/submission_rf-event
echo ""

done

# for model in "bernoulli_nb" "rf" "xgboost"
# do
# python main.py --cross_validate --get_submission --model $model --pick_criteria autothre
# python main.py --cross_validate --get_submission --model $model --pick_criteria autothre --event_wise
# done

# python main.py --cross_validate --get_submission --model rf --pick_criteria autothre --class_weight_scheme customize

# k=1
# for model in "bernoulli_nb" "rf" "xgboost"
# do
# python main.py --cross_validate --get_submission --model $model --pick_k $k
# python main.py --cross_validate --get_submission --model $model --event_wise --pick_k $k
# python main.py --cross_validate --get_submission --model $model --pick_criteria threshold 
# python main.py --cross_validate --get_submission --model $model --event_wise --pick_criteria threshold
# done
# 
# python main.py --cross_validate --get_submission --model rf --pick_k $k --class_weight_scheme customize
# python main.py --cross_validate --get_submission --model rf --pick_criteria threshold --class_weight_scheme customize

# python main.py --cross_validate --use_pca --predict_mode --model xgboost
# python main.py --cross_validate --use_pca --predict_mode --model xgboost --event_wise

