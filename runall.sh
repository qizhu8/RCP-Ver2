python3 test_caseStudy.py
python3 test_competition.py
python3 test_newutility.py
python3 test_caseStudy_w_TCP.py
python3 test_changeCh.py

# utility curve
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/summary/system\ utility_beta.pdf Results/system\ utility-case\ study.pdf
cp Results/case_study_competition_TimeDiscount_alpha_2_0/summary/system\ utility_beta.pdf Results/system\ utility-competition.pdf
cp Results/case_newutility_SumPower_alpha_2_0/summary/system\ utility_beta.pdf Results/system\ utility-newutility.pdf

# retransmission probability with \eta
cp Results/case_study_w_TCP_TimeDiscount_alpha_2_0/summary/retransProb_beta.pdf Results/retransProb_beta-case\ study.pdf

# retransmission probability of RCP over time
cp Results/dynamic_channel_TimeDiscount_alpha_2_0/TimeDiscount_0_3_0/retransProb_overtime.pdf Results/change_of_retransProb_time_RTQ-dynamic\ channel.pdf