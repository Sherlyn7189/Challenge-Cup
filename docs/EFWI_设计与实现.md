# EFWI 璁捐涓庡疄鐜拌鏄?
鏈枃妗ｇ粰鍑轰唬鐮佽儗鍚庣殑鍏紡銆佺鏁ｇ粏鑺傘€佸弬鏁伴€夋嫨鍜岄獙璇佺粨鏋滐紝渚夸簬鎶婂疄鐜拌鏍?涓庝唬鐮侀€愭潯瀵瑰簲銆?
---

## 1. 鐩爣涓庤寖鍥?
杈撳叆锛?
* 浜岀淮寮规€у弬鏁版ā鍨嬶紙`vp, vs, rho`锛屽彲閫?`epsilon, delta, gamma`锛夛紱
* 瑙傛祴绯荤粺锛堢偖鐐广€佹娉㈠櫒銆佸瓙娉€佽褰曢暱搴︼級锛?* 瑙傛祴鐐泦锛堟垨鐢辩湡妯″瀷鍚堟垚锛夈€?
杈撳嚭锛?
* 澶氬昂搴?EFWI 鏇存柊鍚庣殑寮规€фā鍨嬶紱
* 鐩爣鍑芥暟/姊害/鐓ф槑/绾︽潫杩濆弽鍘嗗彶锛?* RTM 淇濆箙鎴愬儚銆佸€捐涓庡弽灏勭偣锛?* 鍙€変笁缁村湴璐ㄦā鍨嬨€?
浠ｇ爜瀹氫綅鏄?*鍙繍琛屻€佸彲楠岃瘉鐨勭爺绌?宸ョ▼鍘熷瀷**锛氭墍鏈夊叧閿楠ら兘鏈夊崟鍏冩祴璇曪紝
姊害鐢ㄦ湁闄愬樊鍒嗘牎楠岋紝姝ｆ紨/浼撮殢鍦ㄧ鏁ｆ剰涔変笅浜掍负杞疆銆?
---

## 2. 寮规€ф尝姝ｆ紨

### 2.1 鏂圭▼

浜岀淮 P-SV 閫熷害鈥斿簲鍔涗竴闃剁郴缁燂細

$$
\begin{aligned}
\rho \dot v_x &= \partial_x \sigma_{xx} + \partial_z \sigma_{xz},\\
\rho \dot v_z &= \partial_x \sigma_{xz} + \partial_z \sigma_{zz},\\
\dot\sigma_{xx} &= C_{11}\partial_x v_x + C_{13}\partial_z v_z,\\
\dot\sigma_{zz} &= C_{13}\partial_x v_x + C_{33}\partial_z v_z,\\
\dot\sigma_{xz} &= C_{44}(\partial_z v_x + \partial_x v_z).
\end{aligned}
$$

鍚勫悜鍚屾€э細

$$
C_{11}=C_{33}=\rho v_p^2,\quad
C_{13}=\rho(v_p^2-2v_s^2),\quad
C_{44}=\rho v_s^2.
$$

VTI锛圱homsen锛夛細

$$
\begin{aligned}
C_{11}&=\rho v_p^2(1+2\epsilon),\\
C_{33}&=\rho v_p^2,\\
C_{44}&=\rho v_s^2,\\
C_{13}&=\rho\left[\sqrt{(v_p^2-v_s^2)(v_p^2(1+2\delta)-v_s^2)}-v_s^2\right].
\end{aligned}
$$

HTI 鍦ㄤ簩缁?x-z 骞抽潰鍐呯敱 VTI 缁?z 杞存棆杞?90掳 寰楀埌锛歚C11 <-> C33`锛宍C13/C44` 涓嶅彉銆?`gamma` 鎺у埗 SH 鍒嗛噺锛屼簩缁?P-SV 姝ｆ紨涓嶄娇鐢紝浣嗕繚鐣欎緵涓夌淮鎵╁睍鍜屾祴浜曡В閲娿€?
### 2.2 浜ら敊缃戞牸

鍦洪噺浣嶇疆锛?
| 鍦?| 浣嶇疆 |
|---|---|
| `vx` | `(i+1/2, j)` |
| `vz` | `(i, j+1/2)` |
| `sxx, szz` | `(i, j)` |
| `sxz` | `(i+1/2, j+1/2)` |

绌洪棿涓€闃跺墠鍚戝樊鍒嗐€佹椂闂翠簩闃讹紱瀵嗗害鍦ㄩ€熷害鐐瑰仛绠楁湳骞冲潎锛宍C44` 鍦ㄨ鐐瑰仛鍥涚偣鍗婂钩鍧囥€?鍥涘懆涓烘捣缁靛惛鏀跺眰锛氶樆灏煎墫闈?
$$
d(s)=d_{\max}\left(\frac{W-s}{W}\right)^2,\quad d_{\max}=\frac{3 v_{\max}}{W h},
$$

姣忔涔樹互 `exp(-d dt)`銆侰FL 妫€鏌ワ細

$$
dt \le 0.5\,\frac{h_{\min}}{v_{\max}\sqrt 2}.
$$

### 2.3 绂绘暎姝ｆ紨鏄犲皠

璁?`s=(sxx,szz,sxz)`銆乣v=(vx,vz)`锛屼竴姝ユ婕斾负

$$
v' = v + \Delta t\,M^{-1}D s,\qquad
s' = s + \Delta t\,C E v',
$$

鍏朵腑 `D` 鏄暎搴︾畻瀛愶紝`E` 鏄簲鍙樼巼绠楀瓙锛宍M=diag(rho)`锛宍C` 鏄垰搴︾煩闃点€?鍔犲叆闃诲凹 `G=diag(exp(-d dt))` 鍚庯細

$$
v' = G(v + \Delta t M^{-1}D s),\quad
s' = G(s + \Delta t C E v').
$$

---

## 3. 浼撮殢鎬佹搴?
### 3.1 绂绘暎杞疆

鐩爣鍑芥暟

$$
J=\frac{1}{2}\frac{\|Pv-d\|^2}{\|d\|^2}.
$$

鍏朵腑 `P` 鏄娉㈠櫒閲囨牱绠楀瓙銆傛暟鎹畫宸?`r=Pv-d` 浣滀负浼撮殢婧愭敞鍏ユ帴鏀剁偣銆?鐢辩 2.3 鑺傜殑姝ｆ紨鏄犲皠锛岀鏁ｄ即闅忥紙鍙嶅悜鏃堕棿锛変负

$$
\tilde v^{n-1/2}=\tilde v^{n+1/2}+\Delta t\,E^T C^T \tilde s^{n+1},
$$

$$
\tilde s^{n}=\tilde s^{n+1}+\Delta t\,D^T M^{-1}\tilde v^{n-1/2}.
$$

娉ㄦ剰绗簩寮忓繀椤荤敤**鍒氭洿鏂拌繃鐨?* `v~^{n-1/2}`锛涜繖鏄鏁ｈ浆缃殑鑰﹀悎椤癸紝鑻ョ敤鏃ч€熷害浼?瀵艰嚧浼撮殢鍦烘寚鏁板闀裤€傞樆灏肩殑杞疆浠嶆槸涔?`G`锛屽苟涓旇鍦ㄦ瘡姝ュ弽鍚戞帹杩?*涔嬪墠**浣滅敤锛?
```
for n = nt-1 ... 0:
    damp(v~, s~)                  # G^T
    inject residual r[n]          # P^T
    accumulate gradients
    v~ = v~ + dt E^T C^T s~
    s~ = s~ + dt D^T M^{-1} v~
```

### 3.2 姊害

$$
\begin{aligned}
g_{C11} &= \sum_n \Delta t\,\tilde\sigma_{xx}\,\dot\varepsilon_{xx},\\
g_{C13} &= \sum_n \Delta t\,(\tilde\sigma_{xx}\dot\varepsilon_{zz}+\tilde\sigma_{zz}\dot\varepsilon_{xx}),\\
g_{C33} &= \sum_n \Delta t\,\tilde\sigma_{zz}\dot\varepsilon_{zz},\\
g_{C44} &= \sum_n \Delta t\,\tilde\sigma_{xz}\dot\varepsilon_{xz},\\
g_\rho &= -\sum_n \Delta t\,\tilde v^{n-1/2}\cdot a^{n+1/2}/\rho,
\end{aligned}
$$

鍏朵腑 `a=(D s + f)/rho` 鏄姞閫熷害銆傚瘑搴﹂」浣跨敤**鏈鎺ㄨ繘鍚庣殑**閫熷害浼撮殢
`v~^{n-1/2}`锛堥€熷害鏇存柊鏂圭▼鐨?Lagrange 涔樺瓙锛夛紝鑰屼笉鏄帹杩涘墠鐨?`v~^{n+1/2}`锛涘悗鑰呬細闅忚褰曢暱搴︾疮绉宸€傚畬鏁存帹瀵艰
[docs/EFWI_姊害鎺ㄥ.md](EFWI_姊害鎺ㄥ.md)銆?
`C44` 鍜?`rho` 鐨勬搴︽寜姝ｆ紨鏃剁殑骞冲潎鏂瑰紡锛堣鐐光啋涓績銆侀€熷害鐐光啋涓績锛夊弽鍚戝垎閰嶏紱
`np.pad(mode="edge")` 鍦ㄨ竟鐣屼骇鐢熺殑閲嶅鑺傜偣浼氭姌鍙犲洖鐗╃悊杈圭晫锛屼繚璇佽竟鐣屽崟鍏?涔熶笌绂绘暎姝ｆ紨涓€鑷淬€?
### 3.3 閾惧紡杞崲

鍚勫悜鍚屾€э細

$$
\begin{aligned}
g_{v_p} &= 2\rho v_p(g_{C11}+g_{C13}+g_{C33}),\\
g_{v_s} &= 2\rho v_s(-2g_{C13}+g_{C44}),\\
g_\rho &= g_\rho^{\text{direct}}
+ v_p^2(g_{C11}+g_{C33})
+ (v_p^2-2v_s^2)g_{C13}
+ v_s^2 g_{C44}.
\end{aligned}
$$

VTI/HTI 鐨?`dC/dp` 瑙?`efwi/gradient.py::_stiffness_derivatives`銆備吉 Hessian
鐢ㄥ悓鏍风殑瀵硅閾惧紡杩戜技锛?
$$
H_p \approx \sum_{C} \left(\frac{\partial C}{\partial p}\right)^2 H_C.
$$

### 3.4 楠岃瘉

`tests/test_gradient.py` 瀵?`vp/vs/rho` 鍚勫彇涓€涓厜婊戞柟鍚戯紝姣旇緝

$$
\frac{J(m+\epsilon d)-J(m-\epsilon d)}{2\epsilon}
\quad\text{vs.}\quad
\sum g_p d.
$$

淇瀵嗗害椤规椂闂撮『搴忎笌杈圭晫骞冲潎鍚庯紝vp/vs 鐩稿璇樊绾?`10^-5`鈥揱10^-6`锛宺ho
绾?`10^-7`鈥揱10^-4`锛坮ho 鐨勯珮鏂柟鍚戝洜璐ㄩ噺椤?閾惧紡椤规帴杩戞姷娑堬紝鐩稿璇樊琚斁澶э紝
浣嗙粷瀵硅宸緢灏忥級銆傝缁嗙粨鏋溿€侀€愰」鍒嗚В鍜屾帴鍙ｈ鏄庤
[docs/EFWI_姊害鎺ㄥ.md](EFWI_姊害鎺ㄥ.md)銆?
鍦ㄥ紑鍙戣繃绋嬩腑杩橀€愮偣姣旇緝浜嗗崟缃戞牸鎵板姩锛岀‘璁ゆ棭鏈熺殑鏈夐檺宸垎姹℃煋鏉ヨ嚜鎵板姩
`vmax` 瀵艰嚧鐨勫惛鏀跺眰鍙樺寲锛涘悗灏嗗惛鏀跺眰鍙傝€冮€熷害鍥哄畾锛屽苟淇瀵嗗害浼撮殢鐨勬椂闂?椤哄簭涓庤竟鐣屾姌鍙狅紝鏂瑰悜瀵兼暟璇樊闄嶅埌 `10^-6` 閲忕骇銆?
---

## 4. 澶氬昂搴﹀弽婕?
`filters.WienerFilter`锛?
1. 瀵瑰弬鑰冪偖闆嗗姞 Tukey 绐椼€佸幓鐩存祦锛?2. 鐢?Butterworth 甯﹂€氭瀯閫犵洰鏍?`y`锛?3. 棰戝煙鏈€灏忎簩涔橈細

$$
H(f)=\frac{P_{xy}(f)}{P_{xx}(f)+\epsilon\max P_{xx}};
$$

4. 骞虫粦 `H`锛屽湪 `[f_low/2^0.5, f_high*2^0.5]` 澶栧姞浣欏鸡鎺╄啘锛?5. 浣滅敤鍒版墍鏈夌偖闆嗭紙闆剁浉浣嶏級銆?
榛樿涓ょ骇锛?
| 棰戝甫 | 杩唬娆℃暟 | 鐩殑 |
|---|---:|---|
| 2鈥? Hz | 25 | 寤虹珛澶у昂搴︽瀯閫狅紝缂撹В鍛ㄨ烦/灞€閮ㄦ瀬灏?|
| 3鈥?5 Hz | 25 | 鍒诲垝缁嗚妭 |

姣忎釜棰戝甫閲嶆柊璁捐婊ゆ尝鍣ㄣ€侀噸缃?L-BFGS锛屽苟鎶婃护娉㈠搷搴斿悓鏃朵綔鐢ㄥ埌闇囨簮瀛愭尝锛?鐢?*甯﹂檺瀛愭尝姝ｆ紨**锛涜繖鏍蜂吉 Hessian 鏄綋鍓嶉甯︾殑鐓ф槑锛岃€屼笉鏄叏棰戝甫鐓ф槑銆?鐩爣鍑芥暟鐢ㄨ娴嬭兘閲忓綊涓€鍖栵紝渚夸簬璺ㄩ甯︽瘮杈冦€傞鐜囬€掕繘涓?Nyquist 鏍￠獙瑙?[docs/EFWI_姊害棰勫鐞嗕笌澶氬昂搴?md](EFWI_姊害棰勫鐞嗕笌澶氬昂搴?md)銆?
---

## 5. 浼?Hessian 鐓ф槑琛ュ伩

鍏?Hessian 鎴愭湰涓嶅彲鎺ュ彈锛涢珮棰?+ 鏃犻檺瀛斿緞杩戜技涓嬪彧淇濈暀鍙傛暟绌洪棿铏氭嫙婧愮殑瀵硅鍏冦€?姝ｆ紨鏃剁疮绉細

$$
H_{C11}=\sum \dot\varepsilon_{xx}^2,\quad
H_{C13}=\sum(\dot\varepsilon_{xx}^2+\dot\varepsilon_{zz}^2),\quad
H_{C33}=\sum \dot\varepsilon_{zz}^2,\quad
H_{C44}=\sum \dot\varepsilon_{xz}^2,
$$

$$
H_\rho^{\text{mass}}=\sum\left[\left(\frac{a_x}{\rho_x}\right)^2+
\left(\frac{a_z}{\rho_z}\right)^2\right],\quad
H_\theta=\sum(\dot\varepsilon_{xx}+\dot\varepsilon_{zz})^2.
$$

鍚勫悜鍚屾€?vp 浣跨敤鐩存帴铏氭嫙婧愶紝鑰屼笉鏄?C 鍒嗛噺鐨勯摼寮忓钩鏂癸細

$$
H_{v_p}=8\rho^2v_p^2H_\theta,\qquad
H_{v_s}=16\rho^2v_s^2(H_{C11}+H_{C33})+4\rho^2v_s^2H_{C44}.
$$

rho 浣跨敤璐ㄩ噺椤?+ 鍒氬害閾惧紡椤癸紱VTI/HTI 浣跨敤 `dC/dp` 鐨勫瑙掗摼寮忚繎浼笺€?
棰勬潯浠讹細

$$
g^{\text{pre}}_p=
\frac{g_p}{H_p+\eta\max H_p+\eta_w\max H_p+h_{\min}},
$$

骞剁敤 `ph_max_gain` 闄愬埗鐩稿涓綅澧炵泭鐨勬斁澶у€嶆暟锛岄潪鏈夐檺鍊兼竻闆躲€?`H` 鐨勯珮鏂钩婊戯紙榛樿 2 涓綉鏍硷級鍙敤浜庣ǔ瀹氬瑙掍及璁★紝涓嶆浛浠ｈ櫄鎷熸簮骞虫柟绉垎銆?鍙傛暟涓庣ǔ瀹氭€ф祴璇曡瑙?[docs/EFWI_姊害棰勫鐞嗕笌澶氬昂搴?md](EFWI_姊害棰勫鐞嗕笌澶氬昂搴?md)銆?
---

## 6. 浼樺寲

* 鍙傛暟鍖栵細`vp/vs/rho` 鐢ㄧ浉瀵瑰鏁?`x=log(m/m_ref)`锛沗epsilon/delta` 绾挎€э紱
* L-BFGS 涓ゅ惊鐜€掑綊锛屽巻鍙查暱搴﹂粯璁?8锛?* 鍒濆 Hessian 鏍囬噺 `gamma=s路y/y路y`锛?* 鏂瑰悜闄愬埗锛歚max_rel_step=0.05`锛堟瘡娆¤凯浠ｆ渶澶х浉瀵瑰弬鏁板彉鍖?5%锛夛紱
* 绾挎悳绱細Armijo + 鎶涚墿绾裤€傜粰瀹?`f0, f(alpha), g0路d`锛屾嫙鍚?
$$
p(a)=f_0+s_0 a+c a^2,\quad
a^*=-\frac{s_0 a^2}{2(f_1-f_0-s_0 a)},
$$

鍦?`[0.1a, 3a]` 鍐呭彇鍊欓€夛紝蹇呰鏃跺洖婧€傜嚎鎼滅储鐨勬枩鐜囩敤**鐪熷疄姊害**
`g_true路d`锛岃€?L-BFGS 鏂瑰悜鐢ㄩ鏉′欢姊害锛岄伩鍏嶉潪涓嬮檷鏂瑰悜銆?
姣忔杩唬鐨勪唬浠风害涓猴細

1. 涓€娆＄嚎鎼滅储锛?鈥? 娆℃婕旓級锛?2. 鍦ㄦ柊妯″瀷涓婇噸鏂拌绠楃洰鏍?姊害+浼?Hessian锛堟婕?浼撮殢锛夈€?
---

## 7. 鍦拌川/宀╃煶鐗╃悊绾︽潫

### 7.1 鍑搁泦

1. **涓婁笅鐣?*锛歚vp鈭圼1200,6500]`銆乣vs鈭圼250,3800]`銆乣rho鈭圼1400,3200]`锛?2. **姊害涓婇檺**锛歚|鈭倄 m|鈮x`銆乣|鈭倆 m|鈮z`锛?3. **Gardner 绠￠亾**锛?
$$
\left|\ln\rho-\ln a-b\ln v_p\right|\le\tau,
$$

鍦?`(ln vp, ln rho)` 绌洪棿鏄嚎鎬у甫锛屽嚫闆嗭紝鎶曞奖鏈夎В鏋愯В锛氭部娉曞悜
`(-b,1)` 绉诲姩銆?
### 7.2 姊害闆嗗悎鐨勬姇褰?
姹傝В

$$
\min_u \frac12\|u-m\|^2\quad
\text{s.t. } |D_x u|\le g_x,\ |D_z u|\le g_z.
$$

ADMM锛?
$$
u=\arg\min \frac12\|u-m\|^2+\frac{\rho}{2}\|D_xu-p_x+q_x\|^2+\frac{\rho}{2}\|D_zu-p_z+q_z\|^2,
$$

$$
(I+\rho D_x^TD_x+\rho D_z^TD_z)u=m+\rho D_x^T(p_x-q_x)+\rho D_z^T(p_z-q_z).
$$

鐢?DCT-II 瀵硅鍖栵紙Neumann 杈圭晫锛夛細

$$
\lambda_x(k)=\left(\frac{2\sin(\pi k/(2N_x))}{\Delta x}\right)^2,
$$

涓€娆?DCT/IDCT 鍗冲彲瑙?u銆傚懆鏈?FFT 浼氬湪妯″瀷杈圭晫鎶婂乏鍙?涓婁笅閿欒缂濆悎锛屽鑷?杈圭晫宸ㄥぇ鍋囨搴︼紱DCT 鐗堟湰淇浜嗚闂銆?
### 7.3 Dykstra 涓?ADMM

Dykstra锛?
```
x = m
p_i = 0  (姣忎釜闆嗗悎鐙珛)
repeat:
  for i:
    y = x + p_i
    z = Proj_i(y)
    p_i = y - z
    x = z
```

鍏抽敭锛?*姣忎釜闆嗗悎缁存姢鐙珛鐨勫閲?`p_i`**锛涙墍鏈夐泦鍚堝叡鐢ㄤ竴涓?`p` 浼氭妸鍏朵粬闆嗗悎
鐨勪慨姝ｉ噸鏂板姞鍥炲幓锛屽鑷存姇褰卞け鏁堛€?
Consensus ADMM 浣滀负澶囬€夛細

$$
m^{k+1}=\frac{m_0+\rho\sum_i(z_i-u_i)}{1+\rho K},\quad
z_i^{k+1}=\mathrm{Proj}_i(m^{k+1}+u_i^k),\quad
u_i^{k+1}=u_i^k+m^{k+1}-z_i^{k+1}.
$$

### 7.4 榛樿鍙傛暟

| 绾︽潫 | 榛樿鍊?|
|---|---|
| vp 姊害涓婇檺 (妯悜, 鍨傚悜) | (8, 12) (m/s)/m |
| vs 姊害涓婇檺 | (5, 8) |
| rho 姊害涓婇檺 | (4, 6) (kg/m鲁)/m |
| Gardner | a=310, b=0.25, tol=0.10 |
| Dykstra 澶栧眰 | 6 |
| 姊害鎶曞奖鍐呭眰 | 20锛坮ho_ADMM=10锛?|

姊害涓婇檺鏄満鍦扮浉鍏冲弬鏁帮紱瀵瑰皷閿愮晫闈㈠簲鏀惧锛屽惁鍒欐姇褰变細鎶婄晫闈㈣繃搴﹀钩婊戙€?
---

## 8. RTM 淇濆箙鎴愬儚

### 8.1 鎴愬儚鏉′欢

瀵规瘡鐐細

1. 姝ｆ紨婧愭尝鍦猴紝淇濆瓨 P锛堟暎搴︼級鎴?S锛堟棆搴︼級锛?2. 鎺ユ敹鏁版嵁浣滀负浼撮殢婧愬弽浼狅紱
3. 浜掔浉鍏筹細

$$
I(x)=\sum_t s(x,t)r(x,t),\qquad
I_{\text{norm}}(x)=\frac{\sum_t s r}{\sum_t s^2+\epsilon}.
$$

榛樿 `source_normalized`锛岃ˉ鍋垮嚑浣?spreading 鍜岀収鏄庝笉鍧囷紝涓嶅仛 AGC锛?淇濈暀鐩稿鎸箙銆侾/S 鍒嗙閬垮厤娉㈠瀷涓叉壈銆?
### 8.2 鐩磋揪娉笌閲囬泦鑴氬嵃

瀹為檯/鍚堟垚鏁版嵁涓洿杈炬尝鑳介噺姣斿弽灏勬尝寮?1鈥? 涓暟閲忕骇銆備唬鐮佹彁渚涳細

* `top_mute`锛氭寜鐐璺濋娴嬬洿杈炬尝鍒版椂锛屽钩婊戝垏闄わ紱
* `depth_taper`锛氬鎴愬儚缁撴灉鍦ㄩ渿婧愰檮杩戝仛娣卞害鍩熷垏闄ゃ€?
婕旂ず涓袱鑰呴兘鐢紝閬垮厤娴呴儴寮哄亣璞℃帺鐩栫湡瀹炵晫闈€?
### 8.3 鍊捐涓庡弽灏勭偣

缁撴瀯寮犻噺锛?
$$
J=\begin{bmatrix}G*(I_x^2)&G*(I_xI_z)\\G*(I_xI_z)&G*(I_z^2)\end{bmatrix},
$$

涓绘柟鍚?`theta=0.5 atan2(2Jxz, Jxx-Jzz)`锛屽弽灏勯潰鍊捐 `dip=theta-pi/2`锛?鎶樺彔鍒?`[-pi/2,pi/2]`銆傜浉骞叉€?
$$
c=\frac{\sqrt{(J_{xx}-J_{zz})^2+4J_{xz}^2}}{J_{xx}+J_{zz}}.
$$

鍙嶅皠鐐癸細`|I|` 鐨勫眬閮ㄦ瀬澶у€?+ 鍒嗕綅鏁伴槇鍊笺€?
---

## 9. 娴嬩簳閫傞厤

### 9.1 LAS

鏃犵涓夋柟渚濊禆鐨?LAS 2.0/3.0 ASCII 瑙ｆ瀽锛屾敮鎸侊細

* `DEPT/DEPTH/MD` 娣卞害锛?* `DTC/DT/AC` 绾垫尝鏃跺樊锛?* `DTS/DTSM/DTSF` 妯尝鏃跺樊锛?* `RHOB/RHOZ/DEN` 瀵嗗害锛?* 绌哄€硷紙榛樿 -999.25锛変笌鍗曚綅鎹㈢畻锛坲s/ft銆乽s/m銆乬/cm鲁銆乲g/m鲁锛夈€?
缂哄け妯尝鐢?`vp/1.8`锛岀己澶卞瘑搴︾敤 Gardner銆?
### 9.2 鍒濆妯″瀷

* 娴嬩簳娈靛唴绾挎€ф彃鍊硷紱
* 娴呬簬/娣变簬娴嬩簳娈电敤鏁翠綋鍘嬪疄瓒嬪娍绾挎€у鎺紱
* 澶氫簳娌?x 绾挎€ф彃鍊硷紝浜曞鍙栨渶杩戜簳锛?* 妯悜/绾靛悜楂樻柉骞虫粦锛?* 淇濊瘉 `vs<0.95 vp`銆佸瘑搴﹀湪鍚堢悊鑼冨洿銆?
### 9.3 灏哄害

`design_scale(vmin, vmax, fmax, ppw=8, cfl=0.45)`锛?
$$
\Delta x=\frac{v_{\min}}{f_{\max}\,N_{ppw}},\qquad
\Delta t=\frac{0.45\Delta x}{v_{\max}\sqrt 2}.
$$

娴嬩簳 10 kHz銆乿p=3000 m/s锛歚dx=0.0375 m`銆乣dt鈮? 渭s`锛涘湴闈㈠湴闇?30 Hz锛?`dx=12.5 m`銆乣dt鈮?.33 ms`銆傜洿鎺ユ贩鐢ㄤ細棰戞暎/涓嶇ǔ瀹氥€?
### 9.4 鍚勫悜寮傛€?
`WellLog.thomsen` 鐢?`vs` 涓庡亣瀹氱殑 SH/SV 姣旓紙榛樿 1.15锛変及璁?`gamma=(vsh虏-vsv虏)/(2vsv虏)`锛屽苟鍙?`epsilon=0.9 gamma`銆乣delta=0.5 gamma`锛?涔熷彲鐩存帴缁欏畾銆傚洓宸濇繁灞傞〉宀?SH/SV 宸?15% 鏃?`gamma鈮?.16`銆?
---

## 10. 闂幆

`ClosedLoopWorkflow`锛?
```
for outer in 1..N:
    EFWI(initial/current model)
    RTM(final model, observed data)
    depth mute + dip/reflector extraction
    3D model = dip-guided strike extrapolation
    center slice -> next initial model
```

涓夌淮寤烘ā瀵规瘡涓?y 鎸夋繁搴﹀钩鍧囧€捐鍋氭í鍚戦敊鍔紝淇濇寔浜岀淮鍓栭潰鐨勬瀯閫犺秼鍔匡紱
杈撳嚭 `.npz` 涓庡彲閫?`.vti`锛圥yVista锛夈€備腑蹇冨墫闈㈠弽棣堢粰涓嬩竴杞?EFWI锛?褰㈡垚鈥滃弽婕旀彁渚涙ā鍨嬨€佹垚鍍忔彁渚涚害鏉熲€濈殑浜掗攣銆?
---

## 11. 楠岃瘉涓庢祴璇?
`python -m unittest discover -s tests -v` 瑕嗙洊锛?
| 娴嬭瘯 | 鍐呭 |
|---|---|
| `test_model` | 鍒氬害銆乂TI/HTI 鏃嬭浆銆丟ardner銆佸钩婊?|
| `test_filters` | Wiener 甯﹂€氬舰鐘躲€侀浂鐩镐綅銆佸甫澶栬“鍑?|
| `test_forward` | 鍚勫悜鍚屾€?鍚勫悜寮傛€?鍔涙簮绋冲畾鎬с€侀浂娈嬪樊 |
| `test_adjoint` | 涓€姝ユ婕?浼撮殢鐨勫唴绉亽绛夊紡锛堢鏁ｈ浆缃級 |
| `test_gradient` | 涓変釜鍙傛暟 脳 涓や釜鏂瑰悜 + 闅忔満瑙傛祴鏁版嵁鐨勬湁闄愬樊鍒嗘柟鍚戝鏁般€侀€愮偖绱Н銆佹畫宸?鐩爣鍑芥暟銆丠 闈炶礋 |
| `test_precondition` | H 闈炶礋銆佽В鏋愰鏉′欢鍏紡銆佸急鐓ф槑绋冲畾鎬с€佺┖闂村鐩婂彉鍖?|
| `test_multiscale` | 棰戝甫鏍￠獙銆乄iener 鍝嶅簲銆侀甯﹁兘閲忋€佷袱闃舵鍙嶆紨鎺ㄨ繘 |
| `scripts/check_gradient.py` | 婕旂ず灏哄害涓婄殑姊害妫€楠屽苟杈撳嚭 JSON |
| `test_anisotropy` | VTI 鐨?epsilon/delta 姊害 vs 鏈夐檺宸垎 |
| `test_optimize` | L-BFGS 涓嬮檷鏂瑰悜銆佹姏鐗╃嚎绾挎悳绱?|
| `test_constraints` | 姊害鎶曞奖銆丟ardner銆丏ykstra/ADMM |
| `test_welllog` | LAS 瑙ｆ瀽銆佸欢鎷撱€佸垵濮嬫ā鍨嬨€佸昂搴?|
| `test_workflow` | 涓夌淮寤烘ā銆佷腑蹇冨墫闈㈠弽棣?|
| `test_pipeline` | EFWI 闄嶄綆 misfit銆丷TM/鍊捐銆乼op mute |

褰撳墠鍏ㄩ儴 46 涓祴璇曢€氳繃銆?
---

## 12. 宸茬煡闄愬埗

1. 浜岀淮 P-SV锛屼笉鍚?SH/涓夌淮缁曞皠锛?2. 娴风坏鍚告敹灞傞潪 CPML锛?3. 闇囨簮瀛愭尝鍥哄畾涓斿凡鐭ワ紱
4. 瀵嗗害鍙傛暟鏁忔劅搴︿綆锛屽缓璁?Gardner 绾︽潫鎴栧垎姝ュ弽婕旓紱
5. 姊害绾︽潫鐨?ADMM 鎶曞奖鏄凯浠ｈ繎浼硷紝寮虹害鏉?灏栭攼鐣岄潰闇€澧炲姞鍐呭眰杩唬锛?6. 涓夌淮寤烘ā鏄蛋鍚戝鎺ㄦ紨绀猴紝涓嶆槸鐪熸鐨勪笁缁村眰浣嶆彃鍊笺€?
