# python3 test_caseStudy.py # replaced by caseStudy_w_TCP
python3 test_caseStudy_w_TCP.py
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/summary/system\ utility_beta.pdf Results/system\ utility-case\ study.pdf
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/summary/system\ utility_beta.png Results/system\ utility-case\ study.png
# retransmission probability with \eta
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/summary/retransProb_beta.pdf Results/retransProb_beta-case\ study.pdf
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/summary/retransProb_beta.png Results/retransProb_beta-case\ study.png


# competing network
python3 test_competition.py
cp Results/case_study_competition_TimeDiscount_alpha_2_0/summary/system\ utility_beta.pdf Results/system\ utility-competition.pdf
cp Results/case_study_competition_TimeDiscount_alpha_2_0/summary/system\ utility_beta.png Results/system\ utility-competition.png


# new utility
python3 test_newutility.py
# utility curve
cp Results/case_newutility_SumPower_alpha_2_0/summary/system\ utility_beta.pdf Results/system\ utility-newutility.pdf
cp Results/case_newutility_SumPower_alpha_2_0/summary/system\ utility_beta.png Results/system\ utility-newutility.png


# change channel service rate 
python3 test_changeCh.py
# retransmission probability of RCP over time
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/TimeDiscount_0_3_0/retransProb_overtime.pdf Results/change_of_retransProb_time_RTQ-dynamic\ channel.pdf
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/TimeDiscount_0_3_0/retransProb_overtime.png Results/change_of_retransProb_time_RTQ-dynamic\ channel.png


# change channel error rate
python3 test_changeErrorRate.py
cp Results/dynamic_channel_error_channel_error_alpha_2_0/summary/system\ utility_pktDropProb.pdf Results/system\ utility-errorrate.pdf
cp Results/dynamic_channel_error_channel_error_alpha_2_0/summary/system\ utility_pktDropProb.png Results/system\ utility-errorrate.png

# change alpha=1
python3 test_changeAlpha1.py
cp Results/dynamic_channel_TimeDiscount_alpha_1_0/summary/system\ utility_beta.pdf Results/system\ utility-alpha1.pdf
cp Results/dynamic_channel_TimeDiscount_alpha_1_0/summary/system\ utility_beta.png Results/system\ utility-alpha1.png