# EFWI 浼?Hessian 棰勬潯浠朵笌澶氬昂搴︿紭鍖?
鏈枃妗ｅ搴斾唬鐮侊細

* `efwi/fd/kernels.py`锛氭婕旀晱鎰熷満銆乣H_theta`銆佸師鐢熷垰搴?瀵嗗害鐓ф槑绱Н锛?* `efwi/gradient.py`锛氬弬鏁扮┖闂翠吉 Hessian銆侀鏉′欢涓庨摼寮忚浆鎹紱
* `efwi/pseudo_hessian.py`锛氫吉 Hessian 璇婃柇/灏佽锛?* `efwi/filters.py`锛歐iener 甯﹂€氥€侀甯︽牎楠屻€佽兘閲忕粺璁★紱
* `efwi/inversion.py`锛氬灏哄害璋冨害銆佸甫闄愬瓙娉€侀鏉′欢 EFWI锛?* `scripts/run_preconditioning_demo.py`锛氬姣斿浘涓庢祴璇曟姤鍛婄敓鎴愩€?
---

## 1. 浼?Hessian 鐨勫畾涔?
鍏?Hessian 涓?
$$
H_{ij}(x)=\sum_s\int_0^T
\left(\frac{\partial d_s}{\partial m_i(x)}\right)^T
\left(\frac{\partial d_s}{\partial m_j(x)}\right)dt,
$$

鐩存帴璁＄畻涓嶅彲琛屻€備吉 Hessian 閲囩敤**楂橀杩戜技 + 鏃犻檺瀛斿緞 + 鍙繚鐣欏瑙掑厓**锛?
$$
\boxed{
H_{pp}(x)=\sum_s\int_0^T
\left\|S_p(x,t)\right\|^2dt
}
$$

鍏朵腑 \(S_p(x,t)\) 鏄弬鏁?\(p\) 鐨?*铏氭嫙婧?鏁ｅ皠婧?*锛氬綋妯″瀷鍙傛暟鍙戠敓
\(\delta p\) 鎵板姩鏃讹紝姝ｆ紨鏂圭▼鍙崇鏂板鐨勬簮椤广€傚畠涓嶆槸瀵规搴﹀仛骞虫粦锛屼篃涓嶆槸
瀵瑰弽婕旂粨鏋滃仛鍥惧儚澶勭悊銆?
### 1.1 鍚勫悜鍚屾€у脊鎬у弬鏁?
璁?
$$
\theta=\dot\varepsilon_{xx}+\dot\varepsilon_{zz},\qquad
\dot\varepsilon_{xz}=\partial_z v_x+\partial_x v_z.
$$

鐢?
$$
C_{11}=C_{33}=\rho v_p^2,\quad
C_{13}=\rho(v_p^2-2v_s^2),\quad
C_{44}=\rho v_s^2
$$

鍙緱锛?
**vp 鐨勮櫄鎷熸簮**锛坰xx銆乻zz 鍚勪竴浠斤級锛?
$$
S_{v_p}^{xx}=S_{v_p}^{zz}=2\rho v_p\theta,\qquad S_{v_p}^{xz}=0,
$$

$$
\boxed{
H_{v_p}=2\int(2\rho v_p\theta)^2dt
=8\rho^2v_p^2\int\theta^2dt
}
$$

浠ｇ爜涓婕旀椂绱Н `H_theta=鈭嘎瞕t`锛坄accumulate_theta_hessian`锛夛紝鍐嶄箻
`8*rho^2*vp^2`銆?
**vs 鐨勮櫄鎷熸簮**锛?
$$
S_{v_s}^{xx}=-4\rho v_s\dot\varepsilon_{zz},\quad
S_{v_s}^{zz}=-4\rho v_s\dot\varepsilon_{xx},\quad
S_{v_s}^{xz}=2\rho v_s\dot\varepsilon_{xz},
$$

$$
\boxed{
H_{v_s}=16\rho^2v_s^2\int(\dot\varepsilon_{xx}^2+\dot\varepsilon_{zz}^2)dt
+4\rho^2v_s^2\int\dot\varepsilon_{xz}^2dt
}
$$

**rho**锛氬瘑搴﹀悓鏃跺嚭鐜板湪璐ㄩ噺椤瑰拰鍒氬害涓€備唬鐮佷娇鐢?
$$
H_\rho=H_\rho^{\text{mass}}
+\left(\frac{\partial C_{11}}{\partial\rho}\right)^2H_{C11}
+\left(\frac{\partial C_{13}}{\partial\rho}\right)^2H_{C13}
+\left(\frac{\partial C_{33}}{\partial\rho}\right)^2H_{C33}
+\left(\frac{\partial C_{44}}{\partial\rho}\right)^2H_{C44},
$$

鍏朵腑

$$
H_\rho^{\text{mass}}
=\int\left[\left(\frac{a_x}{\rho_x}\right)^2+
\left(\frac{a_z}{\rho_z}\right)^2\right]dt,
$$

\(a=(Ds+f)/\rho\) 鏄姞閫熷害銆傝繖鏄瘑搴﹀弬鏁板湪璐ㄩ噺椤?+ 鍒氬害閾惧紡涓嬬殑瀵硅杩戜技銆?
### 1.2 鍚勫悜寮傛€?
VTI/HTI 浣跨敤鍒氬害鍒嗛噺瀵硅鐓ф槑 \(H_{C11},H_{C13},H_{C33},H_{C44}\)锛屽啀鎸?`dC/dp` 閾惧紡骞虫柟锛?
$$
H_p\approx\sum_C\left(\frac{\partial C}{\partial p}\right)^2H_C.
$$

鍚勫悜寮傛€х洰鍓嶄笉鍋?vp 鐨?\(\theta\) 淇锛涜繖鏄凡鐭ョ殑杩戜技锛屾祴璇曚腑鍙
鍚勫悜鍚屾€?vp 浣跨敤鐩存帴铏氭嫙婧愬叕寮忋€?
### 1.3 璁＄畻娴佺▼

姣忎釜姝ｆ紨鏃堕棿姝ワ紙浠?`hessian=True`锛夛細

```
exx, ezz, exz = E v
H_C11 += dt*exx^2
H_C13 += dt*(exx^2+ezz^2)
H_C33 += dt*ezz^2
H_C44 += dt*exz^2
H_rho += dt*((ax/rho_x)^2+(az/rho_z)^2)
H_theta += dt*(exx+ezz)^2
```

鎵€鏈夌偖闆嗘眰鍜屽悗锛岃浆鎹㈠埌 `(vp, vs, rho)`銆俙H` 涓庢搴︾敱**鍚屼竴缁勬婕旀晱鎰熷満**
寰楀埌锛岄伩鍏嶉澶栨婕斻€?
---

## 2. 姊害棰勫鐞?
### 2.1 鍏紡

$$
\boxed{
g_p^{\text{pre}}=
\frac{g_p}{H_p+\eta\max(H_p)+\eta_w\max(H_p)+h_{\min}}
}
$$

鍏朵腑锛?
* \(\eta\)锛歚ph_damping`锛岄粯璁?`1e-3`锛?* \(\eta_w\)锛歚ph_water_level`锛岄粯璁?`0.0`锛堟紨绀轰腑浣跨敤 `1e-3`锛夛紱
* \(h_{\min}\)锛歚ph_min_h`锛岀粷瀵规按灞傦紱
* 瀵?\(H\) 鍋氶珮鏂钩婊戯紙`ph_smooth=2`锛?*鍙敤浜庣ǔ瀹氬瑙掍及璁?*銆侶 鏈韩鐨?  璁＄畻鐢辫櫄鎷熸簮骞虫柟绉垎缁欏嚭锛屼笉鏄€滅畝鍗曞钩婊戞浛浠ｂ€濄€?
### 2.2 鏁板€肩ǔ瀹氭帾鏂?
1. **鐩稿姘村眰**锛歕(\eta\max H\) 闃叉 \(H\to0\) 鏃堕櫎闆讹紱
2. **棰濆姘村眰/缁濆姘村眰**锛氭帶鍒舵繁閮ㄦ瀬寮辩収鏄庯紱
3. **鏈€澶у鐩?cap**锛氫互涓綅澧炵泭涓哄吀鍨嬪€硷紝闄愬埗

$$
\text{gain}\le \text{max\_gain}\times \text{median(gain)},
$$

   榛樿 `ph_max_gain=20`銆傝繖闃叉娣遍儴鍣０琚斁澶?3鈥? 涓暟閲忕骇锛?4. **闈炴湁闄愬€兼竻闆?*锛歚nan_to_num`锛?5. **鍙€夊綊涓€鍖?*锛氶鏉′欢鍚庢寜鍘熷姊害 RMS 褰掍竴鍒板悓涓€閲忕骇锛屼繚鎸?   vp/vs/rho 鍙傛暟闂村昂搴︿竴鑷达紙`ph_normalize=True`锛夈€傚綊涓€鍖栦笉鏀瑰彉绌洪棿
   鐓ф槑琛ュ伩妯″紡锛屽彧鏀瑰彉鏁翠綋灏哄害銆?
### 2.3 棰勬潯浠跺墠鍚庡樊寮?
浠ユ紨绀猴紙81脳61锛? 鐐紝鍒濆妯″瀷锛変负渚嬶細

| stage | param | raw RMS | pre RMS | deep gain | shallow gain | deep boost |
|---|---|---:|---:|---:|---:|---:|
| 2鈥? Hz | vp | 1.403e-06 | 2.028e-07 | 2.48 | 0.144 | 17.2 |
| 2鈥? Hz | vs | 1.320e-06 | 3.317e-07 | 4.89 | 0.250 | 19.5 |
| 2鈥? Hz | rho | 6.018e-07 | 8.971e-07 | 20.4 | 1.485 | 13.7 |
| 3鈥?5 Hz | vp | 1.296e-06 | 3.462e-08 | 0.120 | 0.0267 | 4.50 |
| 3鈥?5 Hz | vs | 1.152e-06 | 5.128e-08 | 0.238 | 0.0445 | 5.34 |
| 3鈥?5 Hz | rho | 5.392e-07 | 8.322e-08 | 0.840 | 0.154 | 5.45 |

锛堟暟鍊肩敱 `scripts/run_preconditioning_demo.py` 鑷姩鍐欏叆 `runs/preconditioning/report.md`锛?鎹㈢綉鏍?鐐泦鍚庝細鍙樺寲銆傦級

`deep gain` 鏄繁閮紙\(z>0.55z_{\max}\)锛夐鏉′欢/鍘熷姊害 RMS 姣旓紱
`deep boost = deep gain / shallow gain` 琛ㄧず鐩稿娴呴儴鐨勭収鏄庤ˉ鍋垮€嶆暟銆傚彲浠ョ湅鍒?棰勬潯浠舵妸娣遍儴鐩稿娴呴儴鐨勬搴﹁兘閲忔彁楂樹簡绾?4.5鈥?9.5 鍊嶏紙闅忛甯?鍙傛暟鍙樺寲锛夈€?
> 瀵规瘮鍥句腑 raw/H/pre 涓夊垪鍚勮嚜鐙珛瑁佸壀鏄剧ず锛屽洜姝や笉鑳界敤鍥剧殑棰滆壊骞呭害鐩存帴
> 姣旇緝缁濆澶у皬锛涚粷瀵规瘮渚嬭 `report.md` 鐨?`scale` 鍒椼€?
---

## 3. 澶氬昂搴︽护娉?
### 3.1 婊ゆ尝鍣ㄧ被鍨?
`WienerFilter` 鏄?*闆剁浉浣?Wiener 甯﹂€?*锛屼笉鏄畝鍗?Butterworth锛?
1. 浠ュ綋鍓嶇偖闆嗕负鍙傝€冿紝鍘荤洿娴併€佸姞 5% Tukey 绐楋紱
2. 鐢?4 闃?Butterworth 甯﹂€氭瀯閫犵悊鎯崇洰鏍?\(y\)锛?3. 棰戝煙鏈€灏忎簩涔?
$$
H(f)=\frac{P_{xy}(f)}{P_{xx}(f)+\epsilon\max P_{xx}},
$$

4. 瀵?\(H\) 鍋?1 Hz 骞虫粦銆侀檺骞?`[0,1.5]`锛?5. 鍦?`[f_low/鈭?, f_high*鈭?]` 澶栧仛浣欏鸡鎺╄啘锛岄槻姝㈠櫔澹伴娈垫斁澶с€?
鍙傛暟锛?
| 鍙傛暟 | 榛樿 | 浣滅敤 |
|---|---:|---|
| `order` | 4 | Butterworth 鐩爣闃舵暟 |
| `eps` | 1e-3 | Wiener 姝ｅ垯鍖?|
| `mask_octaves` | 0.5 | 棰戝甫澶栨帺鑶滆繃娓?|
| `smooth` | 1.0 Hz | 鍝嶅簲骞虫粦 |
| `taper` | 5% | 鏃剁獥锛屾姂鍒跺懆鏈熷欢鎷?|

### 3.2 棰戝甫涓庡瓙娉?
榛樿锛?
| 闃舵 | 棰戝甫 | 杩唬 | 鐩殑 |
|---|---|---:|---|
| `band_8Hz` | 2鈥? Hz | 25 | 寤虹珛澶у昂搴︽瀯閫狅紝缂撹В鍛ㄨ烦 |
| `band_15Hz` | 3鈥?5 Hz | 25 | 鍒诲垝缁嗚妭 |

**甯﹂檺瀛愭尝**锛氭瘡涓樁娈垫妸 Wiener 鍝嶅簲浣滅敤鍒伴渿婧愬瓙娉笂锛屽啀鐢ㄥ甫闄愬瓙娉㈡婕斻€?杩欐牱浼?Hessian 鏄?*褰撳墠棰戝甫鐨勭収鏄?*锛岃€屼笉鏄叏棰戝甫鐓ф槑锛涘湪绾挎€ф护娉㈠亣璁句笅锛?杩欎笌鈥滃鍚堟垚鏁版嵁鍋氬悓甯︽护娉⑩€濈瓑浠凤紝浣?H 鐨勭墿鐞嗗惈涔夋洿鍚堢悊銆?
---

## 4. 澶氬昂搴﹁皟搴︾瓥鐣?
`efwi.filters.validate_stages` 寮哄埗锛?
1. \(0<f_{low}<f_{high}\)锛?2. \(f_{high}\) 鍗曡皟涓嶅噺锛堜粠浣庨鍒伴珮棰戦€掕繘锛夛紱
3. \(f_{high}<0.95f_{Nyquist}\)銆?
璋冨害娴佺▼锛?
```
for stage in stages:
    filt = WienerFilter.design(observed, dt, stage.f_low, stage.f_high)
    stage_wavelet = filt.apply(survey.wavelet)
    obs_f = filt.apply(observed)
    op.set_survey(stage_survey)        # 甯﹂檺瀛愭尝
    reset L-BFGS
    for it in stage.n_iter:
        gradient + pseudo-Hessian (褰撳墠棰戝甫)
        precondition
        L-BFGS + parabolic line search
        constraints
```

* 棰戝甫闂?*淇濈暀妯″瀷**锛坵arm start锛夛紝涓嶉噸缃ā鍨嬶紱
* 姣忎釜棰戝甫閲嶇疆 L-BFGS 鍘嗗彶锛堜笉鍚岄甯︽洸鐜囧樊寮傚ぇ锛夛紱
* 鐩爣鍑芥暟鎸夎娴嬭兘閲忓綊涓€鍖栵紝璺ㄩ甯﹀彲姣旓紱
* 闃舵缁撴灉璁板綍 `stage_summaries`銆乣stage_models`锛屼緵鎶ュ憡/鍥句娇鐢ㄣ€?
鍙嚜瀹氫箟涓夐樁娈碉細

```powershell
.venv\Scripts\python.exe scripts\run_preconditioning_demo.py --stages "2-8,3-12,3-18" --iterations 3
```

---

## 5. 绋冲畾鎬ф祴璇?
`scripts/run_preconditioning_demo.py` 瀵?\(\eta=10^{-8}\dots10^{-1}\) 鎵弿锛?骞跺惎鐢?`max_gain=20`锛?
| param | \(\eta\) | max gain / median gain |
|---|---:|---:|
| vp | 1e-8 | 20.0锛坈ap锛?|
| vp | 1e-5 | 5.12 |
| vp | 1e-3 | 1.04 |
| vp | 1e-1 | 1.00 |
| vs | 1e-8 | 20.0 |
| vs | 1e-3 | 1.07 |
| rho | 1e-8 | 20.0 |
| rho | 1e-3 | 1.10 |

缁撹锛?
* 灏忛樆灏兼椂 cap 鎶婃瀬绔鐩婇檺鍒跺湪 20脳 涓綅澧炵泭锛?* 闃诲凹澧炲ぇ鍚庡鐩婅秼浜?1锛堥鏉′欢閫愭笎閫€鍖栦负鍘熷姊害锛夛紱
* 鎵€鏈夐鏉′欢姊害 `finite=True`锛屾棤 NaN/Inf锛?* `tests/test_precondition.py` 楠岃瘉浜嗏€滄棤 cap 鏃朵弗鏍肩瓑浜庤В鏋愬叕寮忊€濄€?  鈥滈浂 H 杈撳嚭闆舵搴︹€濄€佲€滃急鐓ф槑+鍣０涓嶇垎鐐糕€濄€?
---

## 6. 杈撳嚭

杩愯锛?
```powershell
.venv\Scripts\python.exe scripts\run_preconditioning_demo.py
```

杈撳嚭 `runs/preconditioning/`锛?
```
report.md / report.json              # 璋冨害銆佹搴︾粺璁°€佺ǔ瀹氭€с€侀樁娈?misfit
stage_01_gradient.png               # raw / H / pre 涓夊垪瀵规瘮
stage_02_gradient.png
filter_responses.png                # 鍚勯甯?Wiener 鍝嶅簲
data_spectra.png                    # 鍘熷/鍒嗛甯︽尟骞呰氨
precondition_stability.png          # 闃诲凹-澧炵泭 cap 鏇茬嚎
stage_models.png                    # 鍒濆/鍚勯甯︽ā鍨?鐪熸ā鍨?stage_convergence.png
fwi/model_stage_01.npz ...
```

娴嬭瘯锛?
```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

鏈ā鍧楁柊澧?`tests/test_precondition.py`銆乣tests/test_multiscale.py`锛屽綋鍓?**46 涓祴璇曞叏閮ㄩ€氳繃**锛堝惈姊害銆丷TM銆佹祴浜曘€佸悇鍚戝紓鎬с€侀棴鐜級銆?
---

## 7. 閰嶇疆椤?
| 瀛楁 | 榛樿 | 璇存槑 |
|---|---:|---|
| `FWIConfig.pseudo_hessian` | True | 鍚敤浼?Hessian |
| `ph_damping` | 1e-3 | 鐩稿姘村眰 |
| `ph_water_level` | 0.0 | 棰濆鐩稿姘村眰 |
| `ph_min_h` | 0.0 | 缁濆姘村眰 |
| `ph_max_gain` | 20.0 | 鏈€澶?涓綅澧炵泭姣?|
| `ph_smooth` | 2 | 浠呭钩婊?H 瀵硅浼拌 |
| `ph_normalize` | True | 棰勬潯浠跺悗 RMS 褰掍竴鍖?|
| `StageConfig.f_low/f_high/n_iter` | 鈥?| 棰戝甫涓庤凯浠ｆ暟 |

