for weight in 0.2 1.0 5.0
do

for pick in "threshold"
do

for threshold in 0.7 0.8 0.9
do
# python main.py --class_weight_scheme customize --predict_mode --use_pca --model rf --additional_weight $weight --force_retrain

python main.py --class_weight_scheme customize --get_submission --use_pca --force_retrain --model rf --additional_weight $weight --merge_priority_score simple --pick_criteria $pick --pick_threshold $threshold
echo "python main.py --class_weight_scheme customize --get_submission --use_pca --force_retrain --model rf --additional_weight $weight --merge_priority_score simple --pick_criteria $pick --pick_threshold $threshold"
python evaluate_v3.py --weight $weight
echo ""

python main.py --class_weight_scheme customize --get_submission --use_pca --force_retrain --model rf --additional_weight $weight --merge_priority_score simple --train_regression --pick_criteria $pick --pick_threshold $threshold
echo "python main.py --class_weight_scheme customize --get_submission --use_pca --force_retrain --model rf --additional_weight $weight --merge_priority_score simple --train_regression --pick_criteria $pick --pick_threshold $threshold"
python evaluate_v3.py --weight $weight
echo ""

python main.py --class_weight_scheme customize --get_submission --use_pca --force_retrain --model rf --additional_weight $weight --merge_priority_score advanced --train_regression --advanced_predict_weight 0.0 --pick_criteria $pick --pick_threshold $threshold
echo "python main.py --class_weight_scheme customize --get_submission --use_pca --force_retrain --model rf --additional_weight $weight --merge_priority_score advanced --train_regression --advanced_predict_weight 0.0 --pick_criteria $pick --pick_threshold $threshold"
python evaluate_v3.py --weight $weight
echo ""

python main.py --class_weight_scheme customize --get_submission --use_pca --force_retrain --model rf --additional_weight $weight --merge_priority_score advanced --train_regression --advanced_predict_weight 0.5 --pick_criteria $pick --pick_threshold $threshold
echo "python main.py --class_weight_scheme customize --get_submission --use_pca --force_retrain --model rf --additional_weight $weight --merge_priority_score advanced --train_regression --advanced_predict_weight 0.5 --pick_criteria $pick --pick_threshold $threshold"
python evaluate_v3.py --weight $weight
echo ""

done
done
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

