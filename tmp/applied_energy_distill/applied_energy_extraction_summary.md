# Applied Energy PDF Extraction Summary

## AE01 Ultra-short-term distributed PV power forecasting for virtual power plant considering data-scarce scenarios

- file: `Ultra-short-term distributed PV power forecasting for virtual power plant considering data-scarce scenarios.pdf`
- doi: `10.1016/j.apenergy.2024.123890`
- pages: 16, chars: 69704, quality: good
- bucket: VPP 预测

### Abstract

Accurate forecasting of distributed photovoltaic (DPV) power plays a crucial role in enabling a virtual power plant (VPP) to grasp its internal DPV characteristics and support the development of optimized internal scheduling strategies. However, majority of the existing forecasting methods are developed rely on sufficient power data samples. Power data scarcity caused by site construction, upgrading or limitation of data share in VPP will lead to poor forecasting accuracy of DPV. To conquer this limitation, the adversarial Graph Neural Network based ultra-short-term DPV power forecasting method for VPP considering the scenarios of data-scarcity is proposed. In this method, the forecasting model is developed on the graph structure data of multi-sites, which is constructed from data-rich region and data-scarce region, respectively. Specifically, graph representation learning and adversarial domain adaptation techniques are utilized to learn domain-invariant graph embedding features, which are further incorporated into the spatiotemporal modeling of distributed photovoltaic power data. As a result, the power forecasting accuracy of sites within the data-scarce region is improved by transferring “ forecasting-related knowledge ” from distributed photovoltaic sites of data-rich region with different data distributions and graph structures. The simulation experiment proves that the forecasting accuracy can be further improved by integrating the domain-invariant features into the power forecasting of distributed photovoltaic in the data-scarce region via a real power dataset.

### Baseline / Metric Hits

- Accuracy metrics The Normalized Root Mean Square Error (NRMSE) and Normalized Mean Absolute Error (NMAE) are used to evaluate the forecasting accuracy.
- NRMSE = 1 p max ̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅ ̅̅̅ ̅ 1 T ∑ T t = 1 ( p t ̂p t ) 2 √ √ √ √ (14) NMAE = 1 T × p max ∑ T t = 1 | p t ̂p t | (15) where p t and ̂p t represent the real and forecasting value at time t , respectively .
- The configuration of simulation In this study, to prove the superiority of the proposed method, several other methods are configured as benchmarks for comparison: Method1: The proposed DAGNN-based DPV power forecasting method.
- Various methods are compared as shown in Table 2 , adopting the concept of ablation experiments when establishing the benchmark methods.
- The comparison of methods 2, 3, and 4 allows for an analysis of the influence of source domain data information on the forecasting task of target domain.
- Building upon this, the comparison between method 1 and other methods is employed to demonstrate that extracting domain-invariant features can mitigate the adverse effects of feature shift between the source and target domains, thereby improving the accuracy of target domain forecasting.
- The comparison of forecasting methods The specific metric values of different methods under 5 forecasting scales are displayed in Tables 3 and 4 .
- Compared with method 2, which relies only on the limited training samples in the target domain, the proposed method can achieve 23.06% and 50.55% improvement in NRMSE accuracy under the forecasting time scale of 15 min and 4 h, respectively.

### Operational Hits

- All rights are reserved, including those for text and data mining, AI training, and similar technologies.
- Against this backdrop, distributed photovoltaic (DPV), an effective avenue for the utilization of solar energy resources, has garnered considerable attention from diverse nations and regions due to their inherent advantages, including adaptable installation, cost-effectiveness, environmental cleanliness, and high efficiency [ 4 , 5 ].
- However, the output of PV systems is highly susceptible to weather conditions, characterized by strong randomness, volatility, and uncertainty.
- It achieves efficient grid integration of DPV and ensures the safe and stable operation of the active distribution network by reconciling the grid-DPV conflict.
- Transferring learning There exist multiple DPVs in one region, one part has a long-term operation time and rich data accumulation, with sufficient model training samples, called the source domain D S = { x S i , y S i } n i = 1 .
- This technology has the potential to significantly enhance the accuracy of net load forecasting, facilitating more efficient resource management, ensuring grid stability, and enabling active participation in electricity markets.
- Both the optimization and control of VPP and their participation in electricity market transactions rely on precise net load forecasting to provide essential information support.
- (2) Supporting the scheduling of distributed energy resources.

### Conclusion

and future work are finally highlighted in 5. 2. Problem statement & basic idea For DPV stations situated in regions with limited data availability, DPV power data from other data-rich regions can be employed for transferring forecasting-related knowledge to support forecasting in the target region. However, directly transferring knowledge acquired from the source domain to the target domain can lead to poor forecasting performance in the final target domain. This issue can be attributed to disparities arising from two main aspects: differences in data distribution characteristics and variations in the structure of the DPV graph within the region. (1) The source and target domains are located in regions with different geographic and climatic environments, resulting in distinct power characteristics. Even neighboring regions, especially for DPV, can exhibit different microclimates due to variances in local geomorphology, which in turn lead to differences in power characteristics between regions. The knowledge related to prediction learned from the source domain data cannot be fully applied to the target domain. (2) Across various regions, the presence of varying numbers of DPVs and differences in their output characteristics gives rise to distinct correlation patterns among DPVs, which result in the difference in DPV graph structure. Consequently, the graph structures formed by DPVs exhibit substantial dissimilarities. Directly transferring the relevant knowledge extracted based on the correlation information learned in the source domain to the target domain without considering the complex and differentiated graph topology will lead to ineffective transferring. To visually depict the aforementioned two issues more explicitly, Fig. 2 displays the data characteristics in t

## AE02 Review of virtual power plant operations: Resource coordination and multidimensional interaction

- file: `Review of virtual power plant operations：Resource coordination and multidimensional interaction.pdf`
- doi: `10.1016/j.apenergy.2023.122284`
- pages: 20, chars: 139760, quality: good
- bucket: VPP 综述

### Abstract

Virtual power plants (VPPs) have become an important technological means for large-scale distributed energy resources to participate in the operation of power systems and electricity markets. However, the operation of VPPs is challenged by stochastic resource characteristics, complex control features, heterogeneous information structures, and strategic game behaviors among stakeholders. To clarify the key problems and solutions to these challenges, this article describes the resource coordination problems and multidimensional interaction mecha nism, and it elaborates the overall decision-making process of VPPs. It also discusses different specific opera tional stages that VPPs should attach importance to from three separate perspectives: energy, communication and the market. From each perspective, every section first analyzes the motivation of decision-making, then analyzes the complexity of the problem models, and summarizes the different modeling methods and solving techniques, thus completing a comprehensive review of VPP operation. Furthermore, the article adopts an interdisciplinary approach, utilizing a literature review and technical statistics to capture the multifaceted contributions of decision-making to VPP operations. It delves into the evolving trends of decision-making technology, analyzed from the coupling cyber-physical-social perspective. Finally, the future trajectory of research issues is deliberated.

### Baseline / Metric Hits

- To overcome this challenge, research [ 138 ] uses a mechanism- Table 5 Technical comparison of five market bidding decision models.
- However, with the market operation of the VPP gaining profits, researchers perceive how to facilitate the operation characteristics and regulation performance of the VPP to benchmark the conventional generation units, which has become a key stage to promote sustainable development.

### Operational Hits

- • Reviewed the operation problems and techniques of VPP from interdisciplinary concept.
- • Summarized the intrinsic change and analyzed the evolution trajectory of VPP operation techniques.
- ARTICLE INFO Keywords: Virtual power plants Operational decision making Market operation Coordinated control Communication control ABSTRACT Virtual power plants (VPPs) have become an important technological means for large-scale distributed energy resources to participate in the operation of power systems and electricity markets.
- However, the operation of VPPs is challenged by stochastic resource characteristics, complex control features, heterogeneous information structures, and strategic game behaviors among stakeholders.
- It also discusses different specific opera tional stages that VPPs should attach importance to from three separate perspectives: energy, communication and the market.
- From each perspective, every section first analyzes the motivation of decision-making, then analyzes the complexity of the problem models, and summarizes the different modeling methods and solving techniques, thus completing a comprehensive review of VPP operation.
- Furthermore, the article adopts an interdisciplinary approach, utilizing a literature review and technical statistics to capture the multifaceted contributions of decision-making to VPP operations.
- Due to the intermittent nature of renewables and the randomness of electricity loads, the need for flexible resources [ 1 ], such as peak shaving, ramping, and regulation reserves [ 2 ], is on the rise.

### Conclusion

, a VPP’s operation control decisions can effectively manage DERs at the local level. This can significantly mitigate the grid’s exposure to fluctuations, such as the unpredictability of distributed renewable outputs and unregulated electric vehicle charging, thus facilitating direct market participation by the VPP. However, as the array of internal and external factors influencing VPP operational de cisions continues to broaden, existing research theories and technical models struggle to illustrate how high-dimensional uncertainty impacts the reliable operation of a VPP. Therefore, refining and standardizing the modeling of operation control decisions remain a crucial task. The above technical models and solutions for the energy dispatching of VPPs are summarized as follows in Table 2. For the control means, a VPP can realize the optimal scheduling of DERs through centralized and distributed control methods. Under centralized control, DERs are linked to a centralized entity in the form of a cluster, which formulates scheduling strategy by considering factors such as the operating cost and efficiency [71]. Centralized control has been studied in depth in the field of optimal scheduling. Study [72] proposed a VPP centralized cooperative optimal scheduling strategy based on weighted multiple objective functions. On this basis, to further reduce the calculation cost in practical applications under uncertain conditions, a simplified centralized robust optimization model based on the Fourier-Motzkin method was constructed in literature [73]. Although centralized control can effectively realize the economic scheduling of the entire interconnected system, it poses great challenges to the computing efficiency of VPPs and faces the problem of violating user privacy. To solve the

## AE03 Uncertainties of virtual power plant: Problems and countermeasures

- file: `Uncertainties of virtual power plant：Problems and countermeasures.pdf`
- doi: `10.1016/j.apenergy.2019.01.224`
- pages: 17, chars: 84945, quality: good
- bucket: VPP 不确定性

### Abstract

A virtual power plant (VPP) is a system that integrates several types of power sources, so as to give a reliable and friendly overall power supply. The sources are often a cluster of distributed generation systems with intermittent renewable energies. Uncertainties are the important issues in researches and applications of VPP. In this paper, renewable power, market price and load demand are classified as major factors of uncertainties, and a comprehensive review of these three factors are given. Based on the classification, the detailed mathematical descriptions are summarized. And then, optimization objectives and constraints, which are adopted to improve the running performance of VPP with uncertainties, are summed up systematically. Solution approaches and tools for the optimization are also presented. At last, demonstration projects are introduced to show how uncertainties are handled in practice. This review paper can provide a rational assistance for researchers who focus on VPP.

### Baseline / Metric Hits

- The proposed approach can increase VPP’s profit compared with deterministic approaches without considering uncertainties.
- The modelling and simulation of a benchmark European distribution network are done in the MATLAB environment and the Gurobi solver is used for the VPP optimization.

### Operational Hits

- ARTICLE INFO Keywords: Uncertainty Virtual power plant Renewable power Market price Load demand ABSTRACT A virtual power plant (VPP) is a system that integrates several types of power sources, so as to give a reliable and friendly overall power supply.
- In this paper, renewable power, market price and load demand are classified as major factors of uncertainties, and a comprehensive review of these three factors are given.
- 1, is just a cloud-based distributed power plant that combines the capabilities of heterogeneous distributed energy resources (DER) to increase power generation, as well as to trade or sell electricity on an open market[3].
- Due to the otherness of application purpose and environment, there are significant differences in system structure and operation modes of VPP.
- Typical projects in Europe, such as the EU FENIX project[5], the EU virtual fuel cell power plant project[6], and the German professional virtual power plant project[7], mainly consider the reliable grid connection of the distributed generation (DG) and power market.
- Moreover, the efficiency and safety of the system operation can be improved[10].
- T •Synergy and interactivity: VPP integrates various kinds of DERs in different areas and achieves coordinated dispatching through the connection of DERs and energy management system (EMS).
- In power industry chain, VPP interacts with market participants[11], assists in network management and provides ancillary services[9].

### Conclusion

Uncertainty is an inescapable and significant issue in optimization of VPP. In this paper, an effort is made to classify uncertainties of VPP into three categories, including renewable power generation uncertainty, market price uncertainty and load demand uncertainty. In order to quantize the uncertainties, probability distribution description, possibilistic description and interval description are introduced. Moreover, the components of objective functions and constraints including uncertainties are figured out separately. The advantages and disadvantages of optimal algorithms, which are used to solve optimization problems of VPP, are summarized and compared. After that, the tools and platforms, which support the running of optimal algorithms, containing GAMS, MATLAB and LINGO, are summed up. At last, three demonstration projects are introduced to show how uncertainties are solved in real applications. It should be noticed that uncertainties analyzed in this paper belong to parameter uncertainty. In fact, the main characteristic of VPP is its virtual structure. It means that the members of VPP can be plugged in and plugged out dynamically. The changes of members reflect the flexibility of VPP, but bring the uncertainty of system structure at the same time. This problem is rarely discussed in published papers. Predictably, structural uncertainties of VPP will be attracted much more attention by many researchers from areas of flexible power supply and competitive electricity market in the near future. Fig. 10.Operational framework of EDISON project. Fig. 11.Operational framework of SHD project. S. Yu et al. Applied Energy 239 (2019) 454–470 468

## AE04 A two-step optimization model for virtual power plant participating in spot market based on energy storage power distribution considering comprehensive forecasting error of renewable energy output

- file: `A two-step optimization model for virtual power plant participating in spot market based on energy storage power distribution considering comprehensive forecasting error of renewable energy output.pdf`
- doi: `10.1016/j.apenergy.2024.124234`
- pages: 16, chars: 71858, quality: good
- bucket: VPP 市场/预测误差

### Abstract

As a complement to the medium and long-term market, the spot market plays an important role in maintaining the security and stability of the power grid. However, as spot trading is more proximate to the actual operation of the power system, the virtual power plant (VPP) is exposed to greater volatility in renewable energy output as well as market prices. To enhance its market competitiveness, this paper constructs a two-step optimization model for VPP participation in the spot market. Based on long-term transaction, the energy storage power is distributed based on the comprehensive forecasting error of renewable energy output. This makes the VPP obtain the maximum profit in the day-ahead market. The trading deviation penalty is reduced by adjusting the energy storage operation plan in the real-time market. The conditional value at risk is used to measure the risk of a trading strategy. The profit of VPP participation in the spot market is analyzed under different risk levels, thus providing a basis for decision makers with different risk appetites. The results show that: (1) The sensitivity of the growth rates of E VaR α t and E CVaR α t to risk level shows an opposite trend. And there is an excess of arbitrage power at low risk level. (2) Although the total cost decreases as the risk level increases, the actual profit in the spot market tends to increase and then decrease. The risk level of the VPP should be set at about 0.4 to make full use of energy storage and obtain the maximum market benefit. (3) The energy storage power can be distributed more accu rately according to the forecasting error of renewable energy. This not only enhances its arbitrage ability, but also ensures its regulation ability, thus improving the overall benefit of VPP at different risk levels. 

### Baseline / Metric Hits

- [ 19 ] proposed a two-stage robust optimization approach to improve the profits of VPP in both DAM and RTM, and pointed out that a detailed comparison of the stochastic and the robust approaches might provide more insight about VPP’s perfor mances.

### Operational Hits

- • The ESS power is distributed into reserve power and arbitrage power.
- • Options for VPP with different risk attitudes are analyzed.
- ARTICLE INFO Keywords: Virtual power plant Spot market Comprehensive forecasting error Power distribution ABSTRACT As a complement to the medium and long-term market, the spot market plays an important role in maintaining the security and stability of the power grid.
- However, as spot trading is more proximate to the actual operation of the power system, the virtual power plant (VPP) is exposed to greater volatility in renewable energy output as well as market prices.
- To enhance its market competitiveness, this paper constructs a two-step optimization model for VPP participation in the spot market.
- This makes the VPP obtain the maximum profit in the day-ahead market.
- The trading deviation penalty is reduced by adjusting the energy storage operation plan in the real-time market.
- The conditional value at risk is used to measure the risk of a trading strategy.

### Conclusion

is given in Section 6. 2. ESS power distribution model based on comprehensive forecasting error 2.1. Comprehensive forecasting error model for renewable energy Since the forecasting errors of wind and PV power generation both affect the reserve power of ESS, this paper combines them to construct a comprehensive forecasting error model [ 38 ]. The error is the deviation of the predicted value from the actual value, as shown in Eq. (1) and Eq. (2) . e w t = Q sj , w t Q yc , w t (1) e pv t = Q sj , pv t Q yc , pv t (2) Where, e w t and e pv t are the error values for wind and PV output, respec tively; Q yc , w t / Q yc , w t and Q sj , w t / Q sj , pv t are their predicted and actual values, respectively. The forecasting error usually obeys a probability distribution and has a boundary. It is therefore necessary to normalize the probabilities in that range, as shown in Eq. (3) and Eq. (4) . p w t e w t ) = p w t e w t ) ∫ b a p w t e w t ) de w t , e w t ∈ [ a , b ] (3) p pv t e pv t ) = p pv t ( e pv t ) ∫ d c p pv t ( e pv t ) de pv t , e pv t ∈ [ c , d ] (4) where, p w t e w t ) / p pv t e pv t ) and p w t e w t ) / p pv t e pv t ) are probability density function of wind and PV power forecasting error before and after normalization, respectively. According to the operation characteristics of VPP, wind and PV power forecasting errors have similar impacts on the ESS. Therefore, comprehensive forecasting error e t can be defined as Eq. (5) : e t = e w t + e pv t (5) Since the forecasting error of wind and PV power are two indepen dent random variables, according to the convolution formula, the probability density function of the comprehensive forecasting error p t ( e t ) is as Eq. (6) : p t ( e t ) = ∫ ∞ ∞ p w t e w t ) p pv t e t e pv t ) de t (6) 2.2. Power distributi

## AE05 Distributionally robust comprehensive declaration strategy of virtual power plant participating in the power market considering flexible ramping product and uncertainties

- file: `Distributionally robust comprehensive declaration strategy of virtual power plant participating in the power market considering flexible ramping product and uncertainties.pdf`
- doi: `10.1016/j.apenergy.2023.121133`
- pages: 19, chars: 86967, quality: good
- bucket: VPP 爬坡/不确定性

### Abstract

The volatility and randomness of renewable energy output make the flexible ramping ability demand of power system more urgent, and flexible ramping product (FRP) can effectively improve the flexibility of the power system. Virtual power plant (VPP) can regulate distributed generation, which is conducive to the regulation potential of flexible resources. To solve the problems of FRP and electric energy markets synergy and the wind power output uncertainty faced by the VPP in the declaration process, this paper proposes a comprehensive declaration strategy for the VPP to participate in the power market considering FRP and uncertainties. Firstly, this paper designs a synergistic trading mode covering flexibility resource demand determination and the virtual bidding curve formation for electric energy and FRP markets, in which the virtual bidding curve can reasonably compensate for the unit opportunity cost. Secondly, a comprehensive declaration-dispatching strategy decision- making model for VPP is constructed, and a two-stage distributed robust optimization (DRO) technology is used to deal with the wind power output uncertainty in the model, and flexible resources such as energy storage are used to mitigate energy deviation in the VPP. Finally, simulations implemented on a typical VPP are delivered to show that: 1) Virtual bidding curve realizes the accurate compensation for the units providing FRP. 2) Compared with the single market, VPP can increase the expected profit by 20.44% and reduce wind curtailment cost by 59.68% in the joint declaration of multi-markets.3) VPP can effectively suppress output uncertainty through energy storage system. 4) DRO model has significant advantages in data-driven, less conservative results and stable running time.

### Baseline / Metric Hits

- 2) Compared with the single market, VPP can increase the expected profit by 20.44% and reduce wind curtailment cost by 59.68% in the joint declaration of multi-markets.3) VPP can effectively suppress output uncertainty through energy storage system.
- Compared with the SO considering a single probability distribution and the RO considering the worst distribution of uncertainty, DRO considers a series of probability distributions (ambiguity sets) that can reasonably model uncertainty, and looks for the worst probability distribution, so it can avoid the shortcomings of SO and RO and has better out-of-sample performance [55].
- In addition, the DRO model is found to have good data-driven characteristics, which can significantly reduce the conservative ness of the results and has low computational complexity compared with the SO and RO models.
- In Case 1, the declared electric quantity in the electric energy market is 5175.737 MWh, the quantity in the FRU market is 265.361MWh, and the quantity in the FRD market is 231.559 MWh; In Case 2, the declared electric energy in the electric energy market is 5699.448 MWh, an in crease of 9.19% compared with Case 1.
- 13 (b), compared with Case 1, Case 2 does not need to reserve capacity to participate in the FRP market.
- (2) Deterministic scenario comparison To better reflect the necessity to consider uncertainty for VPP dec larations, this section sets up a deterministic declaration scenario - Case 3, in which the uncertainty of wind power output is not considered.
- As shown in Table 5 , compared with Case1, the benefits of Case3 increased by 3.91%, but the cost increased by 10.11%, resulting in a 23.03% decrease in profits.
- (3) Model comparison analysis This paper proposes a method for dealing with uncertainty in wind power output based on DRO.

### Operational Hits

- • A declaration-dispatching strategy for VPP participating in the electric energy and FRP markets is constructed.
- • The distributionally robust optimization (DRO) technique is used to deal with the uncertainty in VPP.
- • VPP ’ s declaration strategy and cost-benefits in 96 period are examined.
- ARTICLE INFO Keywords: Flexible ramping product (FRP) Virtual power plant (VPP) Synergistic trading mode Comprehensive declaration strategy Distributionally robust optimization (DRO) Uncertainty ABSTRACT The volatility and randomness of renewable energy output make the flexible ramping ability demand of power system more urgent, and flexible ramping product (FRP) can effectively improve the flexibility of the power system.
- To solve the problems of FRP and electric energy markets synergy and the wind power output uncertainty faced by the VPP in the declaration process, this paper proposes a comprehensive declaration strategy for the VPP to participate in the power market considering FRP and uncertainties.
- Firstly, this paper designs a synergistic trading mode covering flexibility resource demand determination and the virtual bidding curve formation for electric energy and FRP markets, in which the virtual bidding curve can reasonably compensate for the unit opportunity cost.
- Secondly, a comprehensive declaration-dispatching strategy decision- making model for VPP is constructed, and a two-stage distributed robust optimization (DRO) technology is used to deal with the wind power output uncertainty in the model, and flexible resources such as energy storage are used to mitigate energy deviation in the VPP.
- 2) Compared with the single market, VPP can increase the expected profit by 20.44% and reduce wind curtailment cost by 59.68% in the joint declaration of multi-markets.3) VPP can effectively suppress output uncertainty through energy storage system.

### Conclusion

and policy implications The high proportion of renewable energy puts forward higher de mand for the power system operation flexibility. How to stimulate and tap the system technical flexibility potential through the market mech anism, straighten out the relationship between flexibility service cost and value is an urgent issue for the power system flexible operation. In addition, in order to tap the LFM flexibility, the VPP can effectively regulate the distributed generation, which is an emerging flexibility adjustment mode. Based on this, this paper designs a synergistic trading mode of electric energy and FRP markets, constructs a comprehensive declaration-dispatching strategy decision-making model for VPP participating in the two markets, and proposes a model solving tech nology based on DRO, which provides a theoretical tool for improving the performance of VPP participating in the power market. The simulation results and further discussions show that: (1) The synergistic trading mode designed in this paper has good applicability to the joint declaration of electric energy and FRP markets. The virtual bidding curve enables the dispatching model to arrange the units ’ reserved capacity according to the economic principle, and reasonably compensate the unit opportunity cost and price the FRP service. (2) The comprehensive declaration strategy of VPP participating in the electric energy and FRP markets in this paper has increased Fig. 17. Comparison of profit results. Fig. 18. Units output in Case 3. Table 5 Units profit comparison (yuan). Case FRP market benefit Energy market benefit C GC C SC C WC B P Case1 38212.3 372,400 329354.11 2500 2086.6 76671.59 Case 3 45726.8 380978.5 356158.45 2996.24 8543.2 59007.41 Z. Yuanyuan et al. Applied Energy 343 (2023) 121133 16 th

## AE06 Highly accurate peak and valley prediction short-term net load forecasting approach based on decomposition for power systems with high PV penetration

- file: `Highly accurate peak and valley prediction short-term net load forecasting approach based on decomposition for power systems with high PV penetration.pdf`
- doi: `10.1016/j.apenergy.2023.120641`
- pages: 13, chars: 77026, quality: good
- bucket: 净负荷预测

### Abstract

[not found]

### Baseline / Metric Hits

- Statistical metrics, Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) were computed to show the model accuracy.
- Difficulty of measuring (i.e., PV being installed behind the meter) and stochastic nature of the PV generation make the net load forecasting most challenging, in comparison to the forecasting of consumers’ demand.
- Section 4 explains the novel hybrid ICEEMDAN- ANN day-ahead forecast model and comparison of results from different algorithms.
- Conventional non-adaptive analysis methods like Fast Fourier Transform (FFT), Discrete Fourier Transform (DFT) and Short-Time Fourier Transform (STFT) provide a good comparison between time and frequency representations of the input signal, but they are not proven to be very accurate in analysing non-stationary and non-linear signals [19,20].
- Feed forward back propagation has shown a minimum RMSE value compared to LSTM and ELMANNN in the shortterm electrical load forecasting of Western Zone in Bangladesh and it is concluded that FFBP is still a viable option for generating accurate load projections in short-term [47].
- To make a fair comparison, suggested ICEEMDAN- ANN model results have been compared with, traditional feed-forward neural network based on back-propagation (BP neural network model), traditional LSTM network model and EMD based decomposition ANN models.
- (8) would contain noise and so, 𝐼𝑀𝐹 1 generated using CEEMDAN tends to contain higher level of noise in comparison to that in 𝐼𝑀𝐹 1 which is produced using ICEEMDAN.
- Comparison between original Net Load and summation of dominant IMFs.

### Operational Hits

- Accurate short-term net load forecasting is essential to ensure reliable and economical operations of a power system.
- Then, net load decomposition outcomes form the inputs of a computationally efficient and accurate ‘‘Long Short-Term Memory-LSTM’’ network algorithm to produce an accurate day-ahead forecasting, which lays out the foundation of day-ahead power dispatch scheduling.
- Australian Energy Market Operator (AEMO) analysis on the shape of the net load curve on the minimum demand day [2].
- This phenomenon imposes a significant valley to peak ramping regulation stress on the conventional generators.
- This undesirable outcome can be alleviated by re-scheduling the daily operations of the generators.
- Such short-term generation scheduling can be based on the net load forecasting of the grid system.
- Such information is usually unavailable or costly to obtain in practice.
- However, being limited to the use of previous hours accumulated generated operation data only for model parameter estimation is a drawback of this study.

### Conclusion

and discusses the future work. 2. State of the art of data analysis and forecasting Traditionally data-analysis methods were performed by applying linear and stationary assumptions to the original signal [18]. Conventional non-adaptive analysis methods like Fast Fourier Transform (FFT), Discrete Fourier Transform (DFT) and Short-Time Fourier Transform (STFT) provide a good comparison between time and frequency representations of the input signal, but they are not proven to be very accurate in analysing non-stationary and non-linear signals [19,20]. Wigner–Ville and wavelet are other widely used non-adaptive analysis techniques that had been derived from Fourier analysis. Due to the fact that they were derived from Fourier analysis, they also suffer from these drawbacks that Fourier transform experienced [21,22]. On the other hand, wavelet analysis is proven to be a promising technique for the decomposition of non-linear signals, however not being capable of analysing non-stationary signals is a major drawback of this [20]. In contrast to fourier transform, Discrete Wavelet Transform (DWT) is a powerful computational tool for analysing a non-stationary signals by decomposing the signal into a set of mutually orthogonal wavelet basis functions. In here, the efficacy of decomposition mostly depends on the carefully selected mother wavelets, unfortunately, which can be difficult to determine in practice. Recognising the shortcomings of the non-adaptive power smoothing methods, adaptive methods such as the techniques based on the empirical mode decomposition (EMD) have been proposed in [20]. EMD is potentially viable for the time, frequency, and energy representation of non-linear, non-stationary signals where, it treats the signal as a ‘‘fast oscillations superimposed on sl

## AE07 Aggregated Net-load Forecasting using Markov-Chain Monte-Carlo Regression and C-vine copula

- file: `Aggregated Net-load Forecasting using Markov-Chain Monte-Carlo Regression and C-vine copula.pdf`
- doi: `10.1016/j.apenergy.2022.120171`
- pages: 17, chars: 66536, quality: good
- bucket: 聚合净负荷

### Abstract

[not found]

### Baseline / Metric Hits

- Comparison between direct and aggregated NLF shows that aggregated NLF produces accurate forecasts [4].
- Net-load forecasts obtained from the proposed model are compared with forecasts obtained from different reference models to show the improvement in forecasting performance.
- (20), 𝑁𝐿𝐹 𝑡 = 𝑁𝐿𝐹 𝑢𝑝 𝑡 + 𝑁𝐿𝐹 𝑑𝑛 𝑡 2 (20) Forecasting performance of the proposed aggregated NLF model can be evaluated using different parameters such as Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
- Also, performance comparison with different reference models can help to show the performance improvement from reference models.
- Mathematical expressions of performance parameters and the details of reference models used for comparison are given in the next section.
- Performance parameters and reference models MAE (MW) and MAPE (%) are used for forecasting performance evaluation [4,34].
- 11 to 14 and Table 9) along with aggregated net-load forecasts for the comparison.
- Preliminary forecasts obtained from the proposed model (PMCMC) are compared with preliminary forecasts obtained from ANN (PANN), SVR (PSVR) [38], and Grey (PGM) models.

### Operational Hits

- This necessitates accurate net-load forecasts for optimum scheduling and flexibility requirement estimations.
- Net- Load Forecasting (NLF) got only little attention in existing literature even though it is essential for optimal generation scheduling and power system flexibility requirement estimations.
- Therefore, system operation planning becomes complex as it involves multiple uncertain variables like load, wind, and solar generation.
- A new variable, called net-load, is introduced in modern power system (very high renewable penetration) operation planning, instead of handling multiple uncertain variables, to reduce complexity [ 3].
- Dispatchable generation units have to be optimally scheduled for net-load and also power system ∗ Corresponding author.
- Therefore, optimal generation scheduling and accurate power system flexibility requirement estimations necessitate accurate net-load forecasts [4,5].
- However, Net-Load Forecasting (NLF) got only a little attention [ 4,13] even though accurate net-load forecasts are essential for optimum scheduling of dispatchable generators [14] and power system flexibility requirement estimations [ 15].
- Standard copula functions (Elliptical and Archimedean) such as Gaussian and Gumbel copulas are widely used for power system uncertainty modelling [23,24].

### Conclusion

Accurate net-load forecasts are essential for optimum scheduling and accurate flexibility requirement estimations in very high renewable penetrated systems. Therefore, this paper proposes a novel probabilistic aggregated very short-term NLF model using Markov Chain Monte Carlo (MCMC) Regression and C-vine copula. The forecasting performance of the proposed NLF model is evaluated by conducting an analysis of data collected from the BPA balancing area. Analysis shows that the proposed aggregated MCMC model shows superior performance over reference models and shows an accuracy of 95.15%. A comparison of the proposed aggregated net-load forecasting model with direct and preliminary net-load forecasting models shows the significance of aggregation of individual load, wind, solar generation Applied Energy 328 (2022) 120171 16 S. Sreekumar et al. Table 9 MAE of aggregated and direct forecasting models (MW). Seasons Interval AANN ASVR AGM GMCMC PMCMC DMCMC AMCMC Spring Morning 249.84 237.9 232.43 231.87 216.7 249.34 167.22 Noon 366.27 335.21 287.43 286.68 292.3 366.73 250.37 Evening 301.37 268.51 263.58 263.31 242.89 301.2 246.11 Night 358.41 335.73 217.9 217.5 217.5 359.41 201.47 Fall Morning 396.44 347.84 324.53 324.97 281.07 395.66 273.38 Noon 318.22 297.82 280.92 279.7 272.56 319.04 264.76 Evening 279.31 273.85 253.26 251.83 237.52 278.71 230.54 Night 335.3 372.04 335.35 336.01 314.85 333.5 305.79 Winter Morning 334.15 319.99 343.03 278.11 277.69 337.77 268.86 Noon 265.23 237.82 202.85 225.44 201.5 254.07 192.93 Evening 328.08 282.01 236.23 244.57 230.28 320.07 227.34 Night 437.38 365.01 308.21 307.84 312.11 437.21 305.95 Summer Morning 317.81 257.21 245.27 245.05 191.9 318.47 189.54 Noon 331.18 264.15 178.18 176.34 250.8 330.41 161.72 Evening 396.59 391.8 266.3 270.63 277.

## AE08 A Transformer-based multimodal-learning framework using sky images for ultra-short-term solar irradiance forecasting

- file: `A Transformer-based multimodal-learning framework using sky images for ultra-short-term solar irradiance forecasting.pdf`
- doi: `10.1016/j.apenergy.2023.121160`
- pages: 19, chars: 88466, quality: good
- bucket: Transformer 光伏/太阳能

### Abstract

The development of solar energy is crucial to combat the global climate change and fossil energy crisis. However, the inherent uncertainty of solar power prevents its large-scale integration into power grids. Although various sky-image-derived modeling methods exist to forecast the variations of solar irradiance, few focus on fully uti lizing the coupling correlations between sky images and historical data to improve the forecasting performance. Therefore, a novel multimodal-learning framework is proposed for forecasting global horizontal irradiance (GHI) in the ultra-short-term. First, the historical and empirically estimated clear-sky GHI are encoded by Informer. Then, the ground-based sky images are transformed into optical flow maps, which can be handled by Vision Transformer. Subsequently, a cross-modality attention method is proposed to explore the coupling correlations between the two modalities. Last, a generative decoder is used to implement multi-step forecasting. The experimental results show that the proposed method achieves a normalized root mean square error (NRMSE) of 4.28% in 10-min-ahead forecasting. Several state-of-the-art methods are also used for comparisons. The exper imental results show that the proposed method outperforms the benchmark methods and exhibits higher ac curacy and robustness in ultra-short-term GHI forecasting.

### Baseline / Metric Hits

- The experimental results show that the proposed method achieves a normalized root mean square error (NRMSE) of 4.28% in 10-min-ahead forecasting.
- Several state-of-the-art methods are also used for comparisons.
- The exper imental results show that the proposed method outperforms the benchmark methods and exhibits higher ac curacy and robustness in ultra-short-term GHI forecasting.
- In addition, the method used an improved attention mechanism (AM) with dynamic region of interest (ROI) for feature augmentation, which achieved a normalized root mean squared error (NRMSE) controlled at 5.57 % on 30 min-ahead (MA).
- The method achieved outstanding forecasting performance by control ling the NRMSE at 5.39 % on 10 MA.
- The obtained re sults are also compared with SOTA studies to verify its effectiveness.
- The remainder of this section contains the data description and preprocessing, performance criteria, introduction of SOTA methods for benchmarks.
- The mean absolute error (MAE), mean absolute percentage error (MAPE), and NRMSE were expressed as fol lows.

### Operational Hits

- However, the inherent uncertainty of solar power prevents its large-scale integration into power grids.
- Being clean, safe and inexhaustible [2] , solar energy provides reliable and economical electricity for resi dences and industries [3] .
- However, the primary issues of the solar power integration lie in its intermittence and uncertain nature [5] , as fluctuating output power significantly hinders the stable and economical operations of power grids [3] .
- In other words, the information with large span within the sky images is difficult to be captured by con volutional operations [40], which hinders the performance of the CNN- based methods on cloudy conditions, especially with significant cloud motion.
- Methodology Both power system operations and energy marketing require accu rate solar forecast results.
- ‖ is the concatena tion operation, and d model represents the output dimension of multi headed sparse attention.
- Therefore, the distilling operation is used to privilege the superior ones with dominating features, which ensures that the feature is more refinable for later decoding [41] .
- [38] , the forward of distilling operation from ( j -1)-th layer into j -th layer is defined as: X t j = MaxPool ( ELU ( Conv1 d ( [ X t j 1 ] spr ))) (6) where [•] spr represents the multiheaded sparse self-attention, which means that the multiheaded sparse attention receives the same Q , K , and V as input.

### Conclusion

. 2. Methodology Both power system operations and energy marketing require accu rate solar forecast results. This section first introduces the overall framework of the proposed method. Then, the encoder modules designed for historical time-series and sky images are introduced, respectively. Later, the cross-modality attention is proposed for sub stantial fusion of encoded features. Finally, the decoder for multi-step forecasting is also introduced. 2.1. Multi-step solar irradiance forecasting framework To fully realize the strengths of multimodal-learning framework to seek correlations between different modalities, input data are recon structed to ensure alignment. Using historical GHI, historical clear-sky GHI and historical sky image sequence with aligned length as L x , a L f -step forecasting method is formulated as: ̂ G L f = [ ̂ G t + 1 , … , ̂ G t + L f ] ⇐ F ( G L x , G ∼ L x , I L x + 1 , { θ } ) ⎧ ⎪ ⎪ ⎪ ⎪ ⎨ ⎪ ⎪ ⎪ ⎪ ⎩ G L x = [ G t L x + 1 , G t L x + 2 , … , G t ] G ∼ L x = [ G ∼ t L x + 1 , G ∼ t L x + 2 , … , G ∼ t ] I L x + 1 = [ I t L x , I t L x + 1 , … , I t ] (1) where ̂ G ∈ R L f denotes the L f -dimensional GHI forecasting results. And G ∈ R L x denotes the historical GHI sequence, and ̃ G ∈ R L x denotes the estimated clear-sky GHI, which are entirely calculated by known pa rameters. Since optical flow calculation is operated on consecutive frames of sky images, additional one frame of sky images is received as input. Therefore, I ∈ R ( L x + 1 )× H × W × 3 denotes RGB-channeled ground- based sky image sequence, where H and W are the height and width of input sky images, respectively. F (•) and { θ } denote the proposed method and training parameters, respectively. And ⇐ represents the optimiza tion process. The proposed multimodal-learning framework

## AE09 Applicability analysis of transformer to wind speed forecasting by a novel deep learning framework with multiple atmospheric variables

- file: `Applicability analysis of transformer to wind speed forecasting by a novel deep learning framework with multiple atmospheric variables.pdf`
- doi: `10.1016/j.apenergy.2023.122155`
- pages: 20, chars: 83287, quality: good
- bucket: Transformer 风速

### Abstract

[not found]

### Baseline / Metric Hits

- Compared with several stateof-the-art transformer-based models and baseline models in AI field, the superior performance of our hybrid framework is observed.
- Interestingly, the transformer exhibits the lowest performance, while the GRU demonstrates the most promising results, with the mean absolute error (MAE) of 0.1356 m/s and 0.1085 m/s for one-step ahead forecasting, respectively.
- Furthermore, the proposed framework is verified as the best method compared with some state-of-the-art algorithms.
- However, in this framework, transformer is verified that it exhibits lower accuracy compared with other temporal forecasting models.
- Section 4 demonstrates the validation of proposed framework and comparison with other state-of-the-art models.
- However, 𝑆𝑢𝑚 aggregator has been verified as the most effective aggregator compared with 𝑀𝑒𝑎𝑛 and 𝑀𝑎𝑥 aggregators [68], due to the more strong message storage capabilities.
- The GRU model offers a less intricate architecture in comparison to LSTM model and RNN model, resulting in computational time savings and better effect on training results, it retains the LSTM immunity to the vanishing gradient problem [31].
- Comparative models To showcase the advantages of the proposed framework, a comparison is conducted with several state-of-the-art algorithms in wind speed forecasting task and baseline algorithms in AI field.

### Operational Hits

- In recent years, driven by rapid economic and population growth, the wind energy generation has captured increasing attention due to the escalating demands for renewable energy [3].
- forecasting results can effectively improve the utilization rate of wind energy resources and mitigate the impact of wind power fluctuations on electricity grid stability, which can achieve economical and efficient operation of wind farms [ 7].
- Consequently, accurate wind speed forecasting is increasingly critical in mitigating costs and risks associated with power supply systems [ 4].
- Lastly, long-term wind speed forecasting (from 1 week to 1 year ahead) aids in scheduling related equipment [6].
- [4] used temporal convolutional networks (TCNs) to forecast wind speed, which exploit the correlation of multiple atmospheric variables (e.g., air temperature, solar radiance, relative humidity, etc.) based on convolutional operation.
- However, this method is not well-suited for capturing pairwise dependencies between multiple variables due to the weight-sharing mechanism of the convolutional operation [60].
- The general GNN layer is expressed as: ℎ(0) 𝑣 = 𝑥𝑣, ∀𝑣 ∈ 𝑉 (15) 𝑚(𝑙) 𝑢𝑣 = 𝛤 (𝑙) 𝑀𝑆𝐺 (ℎ(𝑙−1) 𝑣 , ℎ(𝑙−1) 𝑢 ), ∀(𝑢, 𝑣) ∈ 𝐸 (16) 𝑎(𝑙) 𝑣 = 𝛤 (𝑙) 𝐴𝐺𝐺(𝑚(𝑙) 𝑢𝑣 ∣ 𝑢 ∈ 𝑁(𝑣)), ∀𝑣 ∈ 𝑉 (17) ℎ(𝑙) 𝑣 = 𝛤 (𝑙) 𝑈 𝑃 𝑇(ℎ(𝑙−1) 𝑣 , 𝑎(𝑙) 𝑣 ), ∀𝑣 ∈ 𝑉 (18) where ℎ(𝑙) 𝑣 represents node feature at the 𝑙th layer; 𝛤 (𝑙) 𝑀𝑆𝐺 , 𝛤 (𝑙) 𝐴𝐺𝐺 and 𝛤 (𝑙) 𝑈 𝑃 𝑇 represent the operation of ‘‘message’’, ‘‘aggregate’’ and ‘‘update’’, respectively.
- This function is recursively utilized to update each node’s representation and preserve injectiveness (see Fig.

### Conclusion

. 2. Methodology In this study, the intricate wind speed signal is first decomposed into multiple subsequences with distinct frequencies using VMD. Then, multiple GINs are employed to process, aggregate, and update messages for each frequency domain. This approach restricts GINs to handle messages from specific frequency fields, where the subsequences exhibit greater periodicity and predictable trends, providing an ideal environment for GINs to operate effectively. Through the combination of VMD and GIN, the pairwise dependencies among multiple atmospheric variables at different frequencies are sufficiently captured. Consequently, this framework can be seamlessly integrated into any model designed for capturing temporal features. The functionality of this framework is outlined as follows: 𝑥(𝑑) 𝑣 = (𝑥𝑣)𝑉 𝑀𝐷 (1) ℎ(𝑖) 𝑣 = 𝐺𝐼𝑁 (𝑖)(𝑥(𝑖) 𝑣 ), 𝑖 = 1, 2, 3, … , 𝑑 (2) 𝑊 𝑖𝑛𝑑𝑠𝑝𝑒𝑒𝑑 = 𝑑∑ 𝑖=1 𝑀𝑜𝑑𝑒𝑙 (𝑖)(ℎ(𝑖) 𝑣 ) (3) where 𝑥𝑣 refers to original wind speed signal; 𝑥(𝑑) 𝑣 are subsequences in different frequencies; ℎ(𝑖) 𝑣 represents node feature after GIN model; 𝑀𝑜𝑑𝑒𝑙 represents any temporal model (e.g., LSTM, GRU, transformer, etc.). 2.1. Variational mode decomposition VMD is a data-driven signal decomposition technique that has gained popularity in recent years. The VMD algorithm is designed to decompose a given signal into a set of modes, each representing a specific oscillatory component with varying frequencies and amplitudes. These modes are characterized by their center frequencies and bandwidths, making VMD well-suited for analyzing non-stationary and multi-component signals. The primary idea behind VMD is to solve an optimization problem by finding the modes that minimize the mutual information between the decomposed components. The mutual information is calculated between each m

## AE10 Spatio-temporal wind speed forecasting using graph networks and novel Transformer architectures

- file: `Spatio-temporal wind speed forecasting using graph networks and novel Transformer architectures.pdf`
- doi: `10.1016/j.apenergy.2022.120565`
- pages: 13, chars: 75904, quality: good
- bucket: 图网络/Transformer 风速

### Abstract

[not found]

### Baseline / Metric Hits

- The persistence model is a commonly used benchmark in wind-speed forecasting, where the forecasted values, ̂𝑤𝑠𝑡+1 are simply taken as the last recorded value 𝑤𝑠𝑡, i.e.
- The persistence model was used as a benchmark against which to compare all the other models.
- Even though this is quite a trivial method for making forecasts, the model can achieve fairly accurate results in the short-term and is therefore used as an important baseline to outperform.
- Forecasting error To evaluate the predictive performance of the different models, we start by comparing the mean absolute (MAE) and squared (MSE) errors, given by the following equations: MAE = 1 𝑛 𝑛∑ 𝑖=0 |𝑦𝑖 − ̂ 𝑦𝑖| (11) MSE = 1 𝑛 𝑛∑ 𝑖=0 (𝑦𝑖 − ̂ 𝑦𝑖)2, (12) where 𝑛 is the total number of samples and, ̂ 𝑦, the model predictions, which should be close to the targets 𝑦.
- The ST-LogSparse and ST-Informer performed consistently better than the ST-Transformer model across all forecasting horizons in terms of both MSE and MAE, which showed the potential improvements brought by the ProbSparse and convolutional attention mechanisms for wind forecasting.
- MAE for the single-step forecasts, compared to the persistence model, both showed approximately a five percent improvement in MSE.
- Since MSE penalise large errors more heavily than the MAE metric, it meant that for the single-step forecasts, the persistence model had on average fewer slightly smaller errors, but a larger number of drastically wrong predictions than the ST-LogSparse and ST-Informer models.
- To investigate the physical interpretation of the forecasting results in relation to wind energy production, two additional MAE metrics were computed and provided in Table 3 , which correspond to the estimated errors in kW and kWh.

### Operational Hits

- A challenge with physical models is that they come at a very high computational cost, making them less viable for local short-term forecasting [3].
- The main ingredient of WaveNet is dilated causal convolution, which is a 1D convolutional operation where the causality ensures that the model cannot violate the sequence ordering, while the dilation increases the receptive field by skipping input values with a certain step.
- The multi-head attention (MHA) block in the encoder employs full self-attention, where each attention operation can attend to the full input sequence.
- Since the point-wise attention operation described in Section 3.3 is insensitive to local context, causal 1D-convolution was used to compute keys and queries, instead of point-wise linear transformations.
- Aggregated edge features, ̄ 𝑒′ 𝑗, are computed using an aggregation function, 𝜌𝑒→𝑣, as ̄ 𝑒′ 𝑗 = 𝜌𝑒→𝑣(𝐸′ 𝑗 ), where 𝐸′ 𝑗 = {𝑒′ 𝑖𝑗 |∀𝑖 ∈ 𝑗 }, (6) which could for example be a sum or mean operation.
- In particular, as the first operation of FFT-Attention, FFT is applied to the key, query and value inputs.
- This was despite more advanced decomposition using MDWD, which was initially thought better at extracting trend and periodic components at different frequencies, compared to the simple moving average operation used in the Autoformer.
- For the first metric in Table 3, results were fairly similar to those discussed in Table 2 , but arguably more interpretable, in terms of understanding the consequence of differing predictive performances and potential risks associated with the proposed models.

### Conclusion

In recent years, Transformer-based models have presided over sequence-based deep learning, often superseding recurrent or convolutional models. Nevertheless, research employing these architectures for wind forecasting has been scarce. This study considered different Transformer architectures as the main predictor for spatio-temporal multi-step forecasting, focusing on the LogSparse Transformer, Informer and Autoformer. This is the first time many of these have been applied to wind forecasting and placed in a spatio-temporal setting using GNNs. Additionally, the novel FFTransformer was proposed, which is based on signal decomposition using wavelet transform and an adapted attention mechanism in the frequency domain. Results show that the FFTransformer architecture was very competitive, achieving results on par with the Autoformer-based model for the 1- and 6step forecasts, while significantly outperforming all other models for the longer 24-step forecasts. Even though the vanilla Transformer architecture generally did not yield significant improvements over an MLP model, it was seen that the convolutional attention in the LogSparse Transformer and the ProbSparse Attention of the Informer, were able to slightly improve prediction performance. By estimating the associated prediction errors in kW and kWh, we showed the potential physical effects of different forecasting performances with regards to the power grid, with the FFTransformer model showing an additional 5% improvement over all other models for the 4-h forecasts. Nevertheless, obtaining the powers based on the NREL 5 MW reference turbine, the method was fairly trivial and it would be desirable to further test the different models on real wind power datasets. By removing graph connections in the input data, we show

## AE11 Deep probabilistic solar power forecasting with Transformer and Gaussian process approximation

- file: `Deep probabilistic solar power forecasting with Transformer and Gaussian process approximation.pdf`
- doi: `10.1016/j.apenergy.2025.125294`
- pages: 15, chars: 85847, quality: good
- bucket: 概率 Transformer

### Abstract

[not found]

### Baseline / Metric Hits

- Compared to the commonly used probabilistic forecasting method MC Dropout, our method decreases the CRPS index by 22.6% on the Shenzhen dataset and 39.7% on the Xingtai dataset.
- In Section 3, the proposed method is compared with other probabilistic time series forecasting models on multiple solar power generation datasets.
- Compared with methods based on Bayesian neural networks, this method does not require multiple forward propagations and is therefore more computationally efficient.
- Metrics for deterministic forecasting For the deterministic forecasting tasks, Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) serve as our evaluation metrics.
- The definitions of these metrics are as follows: RMSE = √ 1 𝑛 ∑𝑛 𝑖=1(𝑦𝑖 − ̂ 𝑦𝑖) (18) MAE = 1 𝑛 ∑𝑛 𝑖=1 ∣ 𝑦𝑖 − ̂ 𝑦𝑖 ∣ (19) where 𝑦 denotes the observation value from the datasets and ̂ 𝑦denotes the mean value of the forecast distribution obtained by the probabilistic forecasting framework.
- Metrics for probabilistic forecasting In the probabilistic density forecasting task, the Continuous Ranked Probability Score (CRPS) is employed as a metric to assess the accuracy of the probabilistic forecasting.
- CRPS quantifies the discrepancy between the observed values and the forecast distribution’s cumulative distribution function (CDF).
- It is mathematically represented as follows: CRPS(𝐹 , 𝑦) = ∫ +∞ −∞ (𝐹 ( ̂ 𝑦) − 1(𝑦 ≥ ̂ 𝑦))2𝑑 ̂ 𝑦 (20) where 𝑦 represents the observed value from the datasets, 𝐹 denotes the CDF of the forecast distribution, and 1(⋅) denotes an indicator function that equals 1 if its internal condition is met and 0 otherwise.

### Operational Hits

- However, existing methods often fall short in accuracy and operational efficiency.
- Energy storage devices also rely on data about changes in solar power generation to allocate reserved charging capacity effectively.
- All rights are reserved, including those for text and data mining, AI training, and similar technologies.
- These probabilistic forecasting methods are vital for the efficient, reliable, and cost-effective integration of solar energy into power systems.
- They deliver crucial insights into the variability and uncertainty of solar power generation [11], indispensable for various stakeholders.
- [12] proposed a network partition method based on affinity propagation algorithm that considers photovoltaic uncertainty.
- [13] showed that the probabilistic forecasting of co-located wind and photovoltaic parks improves trading in the day-ahead market.
- [14] proposed a multi-time-scale economic dispatch strategy for a multi-source hybrid power system based on the variable confidence level, where the deterministic constraints are transformed into robust constraints that take the uncertainty into account.

### Conclusion

and discusses future research directions. 2. Methodology In this section, we delineate the architecture of our proposed method for probabilistic forecasting of solar power. Fig. 1 illustrates the framework’s structure and the flow of data within it. This framework is structured into three primary components. The first component processes the input comprising weather, temporal, and solar power generation data. The second component, a deep learning-based model for probabilistic time series forecasting, utilizes the processed data to forecast a probability distribution. The final component applies an inverse transformation to this predictive distribution to obtain the probability distribution of anticipated power generation. 2.1. Problem statement In this paper, we focus on estimating future solar power generation using probabilistic time series forecasting. This entails predicting the probability distribution of solar power generation for a future period, based on historical solar power generation data and other variables (e.g., past weather conditions, temporal information). Let 𝑠𝑡 represent the solar power generated at time 𝑡, we can then express our objective as a conditional distribution 𝑃 (𝐬𝑡0∶𝑇 |𝐬1∶𝑡0 , 𝐰1∶𝑡0 , 𝐮1∶𝑇 ), where 𝐬𝑡0∶𝑇 = [𝑠𝑡0 , 𝑠𝑡0+1, … , 𝑠𝑇 ] and 𝐬1∶𝑡0 = [𝑠1, 𝑠2, … , 𝑠𝑡0 ] represent the solar power generation for future and past time period, respectively. 𝑡0 denotes the initial time point for which 𝑠𝑡 is unknown at the time of prediction, while 𝑇 indicates the furthest time point to be predicted. The covariates 𝐰1∶𝑡0 and 𝐮1∶𝑇 , represent weather conditions at the solar power station’s location and temporal features, respectively. Notably, 𝐰1∶𝑡0 is available only for past time points, whereas 𝐮1∶𝑇 is accessible for all time points. It is worth noting tha

## AE12 Physics-informed reinforcement learning for probabilistic wind power forecasting under extreme events

- file: `Physics-informed reinforcement learning for probabilistic wind power forecasting under extreme events.pdf`
- doi: `10.1016/j.apenergy.2024.124068`
- pages: 12, chars: 51431, quality: good
- bucket: Physics-informed 预测

### Abstract

With wind power penetration increases, accurate and reliable wind power forecasting is becoming gradually critical, and data driven model is a promising solution to implement this task. However, limited by deficient data samples under extreme conditions, scarcity of feature measurements fails to meet the number of training samples required, making the forecasting model exhibits low adaptability. This paper proposes a physics-informed reinforcement learning based method for probabilistic forecasting. Analytical physical expression of wind power output under extreme event is established to construct the error evaluation function for abrupt feature. Deep deterministic policy gradient-based quantile fitting model is then proposed, with abrupt feature embedded as the auxiliary input data for neural network. On this basis, parameter training technique under small data set is proposed to deal with the effect of extreme conditions on correction process of network. It extracts key historical transitions from experience replay pool to establish effective training samples, and model parameters are updated through the small-batch learning strategy to minimize long-term error feedback. Test result on the practical cold wave event of wind farms in China shows effectiveness of the proposed method.

### Baseline / Metric Hits

- In this section, the feedback mechanism for probabilistic forecasting is constructed by introducing Pinball loss function, which enables timely and effective evaluation for the model, with the corresponding error evaluation function shown in Eq.
- Comparison results between different methods Taking WF 1#-6# as an example, the comparison results of reli ability and skill score for different methods under 30 min time scale are shown in Fig.
- Comparison results between different time scales Table 1–4 shows comparison results of reliability and skill score of WF 2#-5# under 30 min and 2 h time scale.
- 6–7 shows the comparison results of reliability and skill score of WF 2# and 5# under 30 min-8 h time scale.
- A comparison of a few statistical models for making quantile wind power forecasts.
- Uncertainty estimation of wind power forecasts: Comparison of probabilistic modelling approaches.

### Operational Hits

- Nevertheless, influenced by complex environmental factors, high intermittent and uncertainty occur [ 1 , 2 ].
- Especially under extreme conditions, operation state of wind turbine is affected, making the characteristic of wind power output changes dramatically.
- Existing forecasting techniques can-not effectively track the climate changes, resulting in significant decrease on accuracy of forecasting results and causing considerable damage on safety and stability operation of power system.
- All rights are reserved, including those for text and data mining, AI training, and similar technologies.
- Research on the impact of extreme weather events on new energy generation and grid operation.
- Analysis of the impact of extreme meteorological weather on new energy operation.
- Uncertainty estimation of wind power forecasts: Comparison of probabilistic modelling approaches.
- Short-term forecasting and uncertainty analysis of wind turbine power based on long short-term memory network and Gaussian mixture model.

### Conclusion

To obtain more accurate wind power forecasting results under extreme events, thus overcoming the challenges posed by the scarcity of feature measurements to the accuracy and adaptability of forecasting models, this paper proposes a physics-informed reinforcement learning based method for probabilistic forecasting. Analytical physical expres sion of wind power output under extreme event is established to construct the error evaluation function for abrupt feature. Deep deter ministic policy gradient-based quantile fitting model is then proposed, with abrupt feature embedded as the auxiliary input data for neural network. On this basis, parameter training technique under small data set is proposed to deal with the effect of extreme conditions on correc tion process of network. It extracts key historical transitions from experience replay pool to establish effective training samples, and model parameters are updated through the small-batch learning strategy to minimize long-term error feedback. Test result on the practical cold wave event of wind farms in China shows effectiveness of the proposed method. It indicates that the agent of reinforcement learning can grad ually adapt to the unknown environment by virtue of the interaction process to actively accumulate feature sample and adjust model pa rameters, thus ensuring the forecasting effect under extreme conditions.

## AE13 Physics-constrained wind power forecasting aligned with probability distributions for noise-resilient deep learning

- file: `Physics-constrained wind power forecasting aligned with probability distributions for noise-resilient deep learning.pdf`
- doi: `10.1016/j.apenergy.2025.125295`
- pages: 13, chars: 64775, quality: good
- bucket: Physics-constrained 预测

### Abstract

Wind power plays a critical role in achieving carbon neutrality as one of the key renewable energy sources. However, accurate wind power forecasting is challenged by high-noise forecast wind speed data, compromising forecast accuracy and robustness. To address this issue, we propose theory-guided (physics-constrained) deep- learning wind power forecasting (TgDPF). TgDPF integrates the domain knowledge of wind power curves, which represent the probability distribution of wind power, with the deep learning model Long Short-Term Memory (LSTM). This integration ensures that the model's output aligns with the probability distribution of the wind power, adhering to physical constraints and enhancing noise resistance. Consequently, TgDPF exem plifies a physics-constrained method. While the probability distribution of wind power is crucial for accurate predictions, effectively utilizing this distribution presents significant challenges, including maintaining model differentiability after embedding the distribution and measuring distribution similarity. To overcome these challenges, TgDPF employs kernel density estimation (KDE) to compute the wind power curve, ensuring the model's differentiability. The discrepancy between the LSTM-generated and actual wind power curves, quantified by the Jensen-Shannon (JS) divergence, is incorporated into the LSTM training process. Compared to the MSE loss-trained LSTM model, TgDPF aligns with the pre-calculated wind power curve, enhancing forecasting reli ability and robustness. Experiments on 25 different wind turbines show that the performance of TgDPF is obviously better than that of LSTM when adding noise of different proportions to wind speed. Specifically, when unbiased high noise N(0, 0.5), N(0, 0.7) is added, TgDPF outperforms the MSE

### Baseline / Metric Hits

- In comparison to the commonly used Kullback-Leibler (KL) divergence, JS divergence exhibits symmetry.
- The MSE loss ensures accurate point predictions, establishing a performance baseline.
- We eval uate LSTM trained solely with MSE loss as a baseline for comparison.
- The blue line represents the actual wind power values, the green line denotes the predicted values gener ated by TgDPF, and the red line represents the predictions from the baseline LSTM.
- Table 6 Wind power forecasting experimental results of autoregressive baselines.
- For comparison, the tables also include the results of LSTM trained solely with MSE loss.
- The results clearly demonstrate that, when augmented with JS loss, each of the alternative models significantly outperforms the baseline approach that relies solely on the data-driven MSE loss.
- Autoregressive baselines with historical wind power data In the domain of time series forecasting, autoregressive models that rely solely on historical data can sometimes achieve satisfactory results [ 56 ].

### Operational Hits

- All rights are reserved, including those for text and data mining, AI training, and similar technologies.
- The demand for day-ahead forecasting is reflected in the market oper ation of wind farms, which is more important for the utilization of wind power [ 5 ].
- Smart energy management system for optimal microgrid economic operation.
- Uncertainty estimation for wind energy conversion by probabilistic wind turbine power curve modelling.
- Deep Ensembles Meets Quantile Regression: Uncertainty-aware Imputation for Time Series 2023.

### Conclusion

[not found]

## AE14 Weather-informed probabilistic forecasting and scenario generation in power systems

- file: `Weather-informed probabilistic forecasting and scenario generation in power systems.pdf`
- doi: `10.1016/j.apenergy.2025.125369`
- pages: 17, chars: 94098, quality: good
- bucket: Weather-informed 场景

### Abstract

[not found]

### Baseline / Metric Hits

- For benchmarking probabilistic solar forecasts, [23] proposed a complete-history persistence ensemble (CH- PeEn) approach that utilizes the entire history of measurements to form empirical distributions, providing a universal reference for skill score computation without assuming a particular distribution form.
- • Systematic analysis of weather data integration in high-dimensional RES forecasting, demonstrating 30%–50% improvement in RMSE at individual asset level and significantly enhanced forecast stability for extended horizons (> 24 hours), particularly crucial for day-ahead market operations.
- • Comprehensive benchmarking and comparison of various time series prediction methodologies, including ARIMA, DeepAR, NLinear, DLinear, and TFT, combined with Gaussian copula.
- Additionally, it includes forecasts generated by NREL’s System Advisor Model (SAM) for the year 2019, which is considered for comparison with the developed models in this study.
- traditional statistical methods and advanced DL-based techniques with and without incorporation of weather data, to provide a comprehensive performance comparison.
- The models include: • A persistence baseline model is implemented, which assumes the forecast equals the most recent observation.
- While simple, persistence models are widely used as baselines in power system forecasting due to their interpretability and effectiveness for short-term predictions.
- Let 1 be the column vector of all ones (with proper dimension), the comparison will be: • individual (ind) level — directly comparing the forecast vectors ̂z(𝑠) 𝑖,𝜏 , 𝑠 ∈ {1, … , 𝑆} and the ground truth z𝑖,𝜏; • space-sum (s-sum) level — comparison after summation over all locations, i.e., comparing 1𝑇 ̂z(𝑠) ⋅, with 1𝑇 z⋅,; • time-sum (t-sum) level — comparison after summation over all time steps, i.e., comparing ̂z(𝑠) ⋅, 1 with z⋅, 1.

### Operational Hits

- This operational uncertainty thus complicates grid stability and optimization efforts [1] which, in turn, motivates the need for advanced forecasting tools.
- In that context, academics and practitioners have increasingly moved from deterministic to probabilistic forecasting, which better captures the uncertainty in load and RES output.
- In addition, probabilistic forecasts provide the ability to sample multiple scenarios, which is essential for risk quantification [3] and uncertainty-aware optimization [4].
- All rights are reserved, including those for text and data mining, AI training, and similar technologies.
- These challenges stem largely from the intermittency, uncertainty, and stochasticity associated with the integration of RES and distributed energy resources into existing power networks [18].
- Such complexities highlights the urgent need to pivot from the deterministic forecasting approaches to uncertainty quantification of forecast errors [19,20].
- Quantile regression excels in providing a comprehensive quantification of uncertainty without the need to assume a predetermined functional form or distribution of forecast errors.
- This methodological flexibility is especially pertinent in addressing the complexities of stochastic power systems, thereby enhancing decision-making processes in critical operations such as risk management, unit commitment, economic dispatch, and optimal decision-making [25,26].

### Conclusion

drawn from the load scenarios apply similarly to wind and solar. Fig. 4 shows the effectiveness of WI-TFT with copula in generating realistic scenarios for individual load zones, wind farms, and solar farms. The copula method improves temporal and spatial correlations, better aligning the generated scenarios with the actual data compared to marginal distributions alone. 8. Conclusion This paper proposed a new method based on combination of TFT and Gaussian copula for high-dimensional probabilistic forecasting in RES. Extensive experiments were conducted to compare different time series forecasting methods on a real-world forecasting problem within the MISO system. The results demonstrated the superiority of the WI-TFT model compared to other statistical and DL-based methods, particularly in terms of predictive accuracy (e.g., NMAE, RMSE) and its ability to capture spatio-temporal dependencies. The integration of weather data as covariates also improved the precision of the models over higher lead times. Additionally, the paper highlighted the efficacy of the Gaussian copula method in restoring spatio-temporal correlations, which is crucial for generating realistic scenarios in RES forecasting. This methodological advancement provides a robust framework for addressing the complexities associated with high-dimensional forecasting problems in RES. Applied Energy 384 (2025) 125369 10 H. Zhang et al. Fig. 3. Comparison of Marginal and Copula Scenarios. Generated scenarios for a load (LRZ1) using (a) marginals (i.e., no inclusion of copula) and (e) with the proposed copula method. (b) The ramps corresponding to the actuals and generated scenarios from marginals in (a). (f) The ramps for the actuals and generated scenarios using copula from (e). (c, d) The generated scenarios 

## AE15 Optimal scheduling of renewable energy microgrids: A robust multi-objective approach with machine learning-based probabilistic forecasting

- file: `Optimal scheduling of renewable energy microgrids：A robust multi-objective approach with machine learning-based probabilistic forecasting.pdf`
- doi: `10.1016/j.apenergy.2024.123548`
- pages: 24, chars: 111796, quality: good
- bucket: 预测到调度

### Abstract

[not found]

### Baseline / Metric Hits

- Operating costs were decreased by 11.5% compared with traditional MPC strategies in uncertain scenarios.
- These include the Mean Absolute Error (MAE), the Root Mean Square Error (RMSE), and the Coefficient of Determination ( 𝑅2) [43].
- The MAE provides a straightforward interpretation by presenting the average deviation from the true value.
- On the other hand, RMSE places more emphasis on significant errors, penalizing them more heavily.
- These metrics are mathematically defined as follow: MAE = 1 𝑁 𝑁∑ 𝑖=1 |𝑦𝑖 − ̂ 𝑦𝑖| (4) RMSE = √√√√ 1 𝑁 𝑁∑ 𝑖=1 (𝑦𝑖 − ̂ 𝑦𝑖)2 (5) 𝑅2 = 1 − ∑𝑁 𝑖=1(𝑦𝑖 − ̂ 𝑦𝑖)2 ∑𝑖 = 1 𝑁 (𝑦𝑖 − ̄ 𝑦)2 , (6) where 𝑦𝑖, ̂ 𝑦𝑖, 𝑦𝑖, and 𝑁 are the true value, predicted value, mean value, and number of samples, respectively.
- On the other hand, to assess probabilistic forecasting, the Continuous Ranked Probability Score (CRPS) emerges as a significant measure.
- The CRPS takes into account the predicted value’s cumulative distribution function and the unit step function, 𝐻(𝑦𝑖, 𝑦), where 𝑦 is the actual value [44].
- Its mathematical representation is given by: CRPS = 1 𝑁 𝑁∑ 𝑖=1 ∫ ∞ −∞ [𝐹 (𝑦𝑖) − 𝐻(𝑦𝑖 − ̂ 𝑦𝑖)]2𝑑𝑦𝑖 (7) 2.5.

### Operational Hits

- All rights are reserved, including those for text and data mining, AI training, and similar technologies.
- Contents lists available at ScienceDirect Applied Energy journal homepage: www.elsevier.com/locate/apenergy Optimal scheduling of renewable energy microgrids: A robust multi-objective approach with machine learning-based probabilistic forecasting Diego Aguilar a,∗, Jhon J.
- Moreover, traditional scheduling strategies often overlook the decay in prediction accuracy over time and lack a mechanism for establishing optimal forecasting horizons.
- This research addresses this gap by merging machine learning (ML) probabilistic forecasting with robust optimization to create optimal dispatch schedules for RES MGs.
- It reveals that longer scheduling horizons can reduce dispatch costs but at the expense of forecast accuracy due to increased prediction accuracy decay (PAD).
- To address this, we propose a novel method that determines how to split the time horizon into timeblocks to minimize dispatch costs and maximize forecast accuracy.
- Results offer Pareto-optimal fronts, showing the trade-offs between cost and accuracy at varying confidence levels.
- Solar power proved more cost-effective than wind power due to lower variability, despite wind’s higher energy output.

### Conclusion

and final remarks In this study we use probabilistic forecasting to improve the operations of a RES MG under uncertainty. Predictions are obtained with a LSTM network integrated within a Seq2Seq architecture providing enhanced forecasts while considering uncertain factors such as wind turbulence and the dynamics of the electric markets. The length of execution/prediction horizons in a MG scheduling strategy has a direct impact on the costs and reliability of the system. Longer horizons tend to yield better costs at the expense of lower forecast accuracy, and vice-versa. This is due to the increase in prediction accuracy decay (PAD) over longer time-spans. To address this, we developed an innovative model that subdivides the scheduling horizon into non-uniform timeblocks for optimal trade-offs between cost and accuracy; this smaller execution horizons then serve as the basis for an optimal RoHS. To strengthen the robustness of the method, both deterministic and probabilistic forecasts where used to solve the EDUC on a sample system. The resulting cost-accuracy trade-offs where then analyzed. After adjusting for conservatism, the probabilistic approach may incur Applied Energy 369 (2024) 123548 20 D. Aguilar et al. Fig. 21. Deterministic EDUC results for lowest cost (highest error) scenario. The power transactions of the DERs, grid and load are presented on a 24-h discretized timeline. The negative kW values represent charging power for the ESS (green) and power sold back to the grid (gray). The execution horizons (timeblocks) are also indicated. Fig. 22. Probabilistic (95%) EDUC results for lowest cost (highest error) scenario. The power transactions of the DERs, grid and load are presented on a 24-h discretized timeline. The uncertain intervals for load demand, wind and
