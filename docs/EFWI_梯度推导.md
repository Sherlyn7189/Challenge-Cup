# EFWI 浼撮殢婧愭瀯閫犱笌姊害璁＄畻

鏈枃妗ｅ搴斾唬鐮侊細

* `efwi/fd/kernels.py`锛氭婕斻€佷即闅忋€佹搴︾疮绉牳锛?* `efwi/fd/operator.py`锛歚Elastic2DOperator` 姝ｆ紨/鍗曠偖姊害/澶氱偖绱Н锛?* `efwi/adjoint.py`锛氭畫宸€佺洰鏍囧嚱鏁般€佷即闅忔簮銆佹搴︾疮绉笌鏈夐檺宸垎妫€楠?API锛?* `efwi/gradient.py`锛氬垰搴︽搴﹀埌 vp/vs/rho/鍚勫悜寮傛€у弬鏁扮殑閾惧紡杞崲銆?
---

## 1. 绂绘暎姝ｆ紨涓庣鍙?
浜岀淮 P-SV 閫熷害鈥斿簲鍔涙柟绋嬶紙涓庢婕斾唬鐮佸畬鍏ㄤ竴鑷达級锛?
$$
\begin{aligned}
\rho \dot v_x &= \partial_x \sigma_{xx}+\partial_z \sigma_{xz},\\
\rho \dot v_z &= \partial_x \sigma_{xz}+\partial_z \sigma_{zz},\\
\dot\sigma_{xx} &= C_{11}\partial_x v_x+C_{13}\partial_z v_z,\\
\dot\sigma_{zz} &= C_{13}\partial_x v_x+C_{33}\partial_z v_z,\\
\dot\sigma_{xz} &= C_{44}(\partial_z v_x+\partial_x v_z).
\end{aligned}
$$

璁?$v=(v_x,v_z)$銆?s=(\sigma_{xx},\sigma_{zz},\sigma_{xz})$锛屼氦閿欑綉鏍肩鏁ｅ悗涓€姝ヤ负

$$
\boxed{
v^{n+1/2}=v^{n-1/2}+\Delta t\,M^{-1}\left(Ds^n+f^n\right),\qquad
s^{n+1}=s^n+\Delta t\,C E v^{n+1/2}
}
$$

鍏朵腑锛?
* $D$锛氬簲鍔涙暎搴︾畻瀛愶紙涓績/瑙掔偣 鈫?閫熷害鐐癸級锛?* $E$锛氬簲鍙樼巼绠楀瓙锛堥€熷害鐐?鈫?涓績/瑙掔偣锛夛紱
* $M=\operatorname{diag}(\rho_x,\rho_z)$锛氬瘑搴﹁川閲忕煩闃碉紱
* $C$锛氬垰搴︾煩闃碉紙鍚勫悜鍚屾€ф垨 VTI/HTI锛夛紱
* $f^n$锛氶渿婧愰」锛岀垎鐐告簮娉ㄥ叆 $s$锛屽姏婧愭敞鍏?$v$銆?
鏁版嵁 $d^n=P v^{n+1/2}$锛?P$ 涓烘娉㈢偣鍙岀嚎鎬ч噰鏍风畻瀛愩€傝褰曞悗涔樺惛鏀跺眰闃诲凹
$G=\operatorname{diag}(e^{-d\Delta t})$銆?
---

## 2. 娈嬪樊涓庣洰鏍囧嚱鏁?
鍚堟垚鏁版嵁 $d^{syn}$銆佽娴嬫暟鎹?$d^{obs}$锛?
$$
r=d^{syn}-d^{obs},\qquad
E_{obs}=\sum_{s}\|d_s^{obs}\|^2,
$$

$$
\boxed{
J=\frac{1}{2}\frac{\|r\|^2}{E_{obs}}
=\frac{1}{2E_{obs}}\sum_{s}\sum_{n,r,c}\left(d^{syn}-d^{obs}\right)^2
}
$$

浠ｇ爜锛?
```python
from efwi.adjoint import data_residual, objective

res = data_residual(synthetic, observed)   # r = syn - obs, energy = ||obs||^2
J = objective(res)                         # 0.5 * ||r||^2 / energy
```

`ShotGradient` 淇濆瓨姣忕偖鐨?*鏈綊涓€鍖?* $0.5\|r_s\|^2$锛屾渶鍚庣粺涓€闄や互鍏ㄥ眬
$E_{obs}$锛岄伩鍏嶉€愮偖褰掍竴鍖栨敼鍙樼偖闂存潈閲嶃€?
---

## 3. 浼撮殢婧?
瀵?$J=\frac{1}{2E_{obs}}\|Pv-d\|^2$ 鏈?
$$
\boxed{\frac{\partial J}{\partial v^{n+1/2}}=\frac{1}{E_{obs}}P^T r^n}
$$

鏁版嵁绌洪棿浼撮殢婧愬氨鏄?$P^T r^n$銆備唬鐮佷腑锛?
1. `adjoint_source(residual, gain=...)` 杩斿洖鏁版嵁绌洪棿鐨?$r$锛堝彲閫夊鐩婏級锛?2. `kernels.inject_adjoint_source(...)` 鐢ㄤ笌姝ｆ紨 `record_data` **瀹屽叏鐩稿悓**鐨?   鍙岀嚎鎬ф潈閲嶏紝鎶?$r$ 鏁ｅ姞鍒颁即闅忛€熷害鍦恒€傚洜姝?$P^T$ 鏄?$P$ 鐨勪弗鏍艰浆缃紱
3. 鑻ョ洰鏍囧嚱鏁板凡鍦?`compute_gradient` 涓寜 $E_{obs}$ 褰掍竴鍖栵紝浼撮殢婧愭湰韬?   涓嶅繀鍐嶄箻 $1/E_{obs}$锛屾渶缁堟搴︾粺涓€闄?$E_{obs}$ 鍗冲彲銆?
瀹炵幇浣嶇疆锛歚Elastic2DOperator._shot_gradient`銆?
---

## 4. 浼撮殢娉㈠満

### 4.1 鍙嶅悜閫掓帹

灏嗘婕斾袱姝ョ殑杞疆鎸?*閫嗗簭**浣滅敤锛屽緱鍒板疄鐜伴噰鐢ㄧ殑绂绘暎浼撮殢锛?
$$
\boxed{
\tilde v^{n-1/2}=\tilde v^{n+1/2}+\Delta t\,E^T C^T\tilde s^{n+1},
\qquad
\tilde s^n=\tilde s^{n+1}+\Delta t\,D^T M^{-1}\tilde v^{n-1/2}
}
$$

娉ㄦ剰锛?
1. 鍏堟敞鍏ユ畫宸細
   $\tilde v^{n+1/2}\leftarrow\tilde v^{n+1/2}+P^T r^n$锛?2. $\tilde s^n$ 蹇呴』鐢?*鍒氭洿鏂板畬鐨?* $\tilde v^{n-1/2}$锛岃€屼笉鏄棫鐨?   $\tilde v^{n+1/2}$锛涜繖瀵瑰簲瀹屾暣 Jacobian 涓殑
   $\Delta t^2 C E M^{-1}D$ 鑰﹀悎椤癸紱
3. 姝ｅ悜姣忔鏈熬涔?$G$锛屽弽鍚戝湪姣忔鎺ㄨ繘鍓嶅厛涔?$G$锛?G^T=G$锛夈€?
浠ｇ爜 `kernels.adjoint_step` 鐢ㄥ弻缂撳啿瀹炵幇锛氱涓€閬嶇畻鏂伴€熷害浼撮殢锛岀浜岄亶绠楁柊
搴斿姏浼撮殢锛屾渶鍚庢竻闆惰竟鐣屻€俙tests/test_adjoint.py` 鐩存帴鐢ㄩ殢鏈哄満楠岃瘉

$$
\langle F x,y\rangle=\langle x,F^T y\rangle .
$$

### 4.2 瀹屽叏鍖归厤灞?娴风坏灞?
闃诲凹鏄瑙掔畻瀛愶紝杞疆浠嶄负涔?$G$銆傜敱浜庢婕斿湪璁板綍涔嬪悗闃诲凹锛屽弽鍚戝繀椤诲厛
鈥滆В闃诲凹鈥濓紙浠嶄箻 $G$锛夊啀娉ㄥ叆娈嬪樊/绱姞姊害锛屽惁鍒欎即闅忔柟绋嬩笌绂绘暎姝ｆ紨涓嶄竴鑷淬€?
---

## 5. 姊害琛ㄨ揪寮?
### 5.1 鍒氬害姊害

$$
\boxed{
\begin{aligned}
g_{C11}&=\sum_n \Delta t\,\tilde\sigma_{xx}^{n+1}\,\dot\varepsilon_{xx}^{n},\\
g_{C13}&=\sum_n \Delta t\left(\tilde\sigma_{xx}^{n+1}\dot\varepsilon_{zz}^{n}
+\tilde\sigma_{zz}^{n+1}\dot\varepsilon_{xx}^{n}\right),\\
g_{C33}&=\sum_n \Delta t\,\tilde\sigma_{zz}^{n+1}\,\dot\varepsilon_{zz}^{n},\\
g_{C44}&=\sum_n \Delta t\,\tilde\sigma_{xz}^{n+1}\,\dot\varepsilon_{xz}^{n}.
\end{aligned}}
$$

杩欓噷 $\tilde s^{n+1}$ 鏄?*鏈瀵瑰簲鐨勪即闅忓簲鍔?*锛堜唬鐮佷腑鍦?`adjoint_step` 涔嬪墠鐨?`saxx/sazz/saxz`锛夛紝$\dot\varepsilon^n$ 鏄婕斾繚瀛樼殑搴斿彉鐜囥€?C_{44}$ 浣嶄簬
瑙掔偣锛屽弽浼犳椂鎸夋婕旂殑 $1/4$ 骞冲潎鍒嗛厤鍒板洓涓腑蹇冦€?
### 5.2 瀵嗗害姊害

瀵嗗害鍙嚭鐜板湪鍔ㄩ噺鏂圭▼鐨勮川閲忛」锛?
$$
\frac{\partial v^{n+1/2}}{\partial \rho}
=-\Delta t\,\frac{Ds^n+f^n}{\rho^2}
=-\Delta t\,\frac{a^{n+1/2}}{\rho}.
$$

鍥犳

$$
\boxed{
g_\rho^{dir}=-\sum_n \Delta t\,\tilde v^{n-1/2}\cdot
\frac{a^{n+1/2}}{\rho}
}
$$

**鍏抽敭缁嗚妭锛氫娇鐢?$\tilde v^{n-1/2}$锛堟湰姝ユ帹杩涘悗鐨勯€熷害浼撮殢锛夛紝鑰屼笉鏄?$\tilde v^{n+1/2}$銆?* 瀹冩槸閫熷害鏇存柊鏂圭▼鐨?Lagrange 涔樺瓙銆傝嫢璇敤鎺ㄨ繘鍓嶇殑
$\tilde v^{n+1/2}$锛岀煭璁板綍鏃惰宸緢灏忥紝浣嗛殢璁板綍闀垮害绱Н锛屾祴璇曚腑 nt=30 鏃?瀵嗗害鏂瑰悜瀵兼暟浼氬亸澶х害 6%锛涗慨姝ｅ悗璇樊闄嶅埌 $10^{-7}$ 閲忕骇锛堣绗?8 鑺傦級銆?
浠ｇ爜锛?
```python
kernels.adjoint_stiffness_gradient(...)   # 鍦?adjoint_step 涔嬪墠
kernels.adjoint_step(...)                 # -> vax2, vaz2 (v~^{n-1/2})
kernels.adjoint_density_gradient(vax2, vaz2, ax, az, rho_x, rho_z, ...)
```

`rho_x/rho_z` 鏄婕旀椂鐨勭畻鏈钩鍧囷紱鍙嶄紶鏃跺悇鍒?$1/2$锛岃竟鐣岄噸澶嶈妭鐐规寜
`np.pad(mode="edge")` 鐨勭瓑浠峰叧绯绘姌鍙犲洖鐗╃悊杈圭晫銆?
### 5.3 閾惧紡杞崲鍒?vp/vs/rho

鍚勫悜鍚屾€э細

$$
\begin{aligned}
g_{v_p}&=2\rho v_p\left(g_{C11}+g_{C13}+g_{C33}\right),\\
g_{v_s}&=2\rho v_s\left(-2g_{C13}+g_{C44}\right),\\
g_\rho&=g_\rho^{dir}
+v_p^2\left(g_{C11}+g_{C33}\right)
+\left(v_p^2-2v_s^2\right)g_{C13}
+v_s^2g_{C44}.
\end{aligned}
$$

VTI/HTI 鐨?$C_{ij}(v_p,v_s,\rho,\epsilon,\delta)$ 鍋忓瑙?`efwi/gradient.py::_stiffness_derivatives`锛涜嫢 `invert_anisotropy=True`锛?棰濆缁欏嚭

$$
g_\epsilon = 2\rho v_p^2\,g_{C11}\ \text{(VTI)},\qquad
g_\delta = \rho\frac{\partial\sqrt{Q}}{\partial\delta}g_{C13}.
$$

---

## 6. 澶氱偖绱Н涓庡綊涓€鍖?
瀵规瘡鐐嫭绔嬭绠楀師濮嬫搴?$g_p^{(s)}$锛岀劧鍚?
$$
\boxed{
g_p=\frac{1}{E_{obs}}\sum_{s=1}^{N_s}g_p^{(s)},\qquad
J=\frac{1}{E_{obs}}\sum_{s=1}^{N_s}\frac12\|r_s\|^2
}
$$

浼?Hessian 涔熸寜鐐眰鍜岋紙涓嶅仛鑳介噺褰掍竴鍖栵紝鐢ㄤ簬鐩稿鐓ф槑琛ュ伩锛夈€俙AdjointResult.g`
宸叉寜 $E_{obs}$ 褰掍竴鍖栵紝`AdjointResult.h` 淇濈暀鍘熷鐓ф槑閲忕翰銆?
浠ｇ爜鎺ュ彛锛?
```python
from efwi.adjoint import ShotGradientAccumulator, compute_adjoint_gradients

result = compute_adjoint_gradients(op, model, observed)   # vp/vs/rho
acc = ShotGradientAccumulator(model, ("vp", "vs", "rho"), obs_energy=E)
for s in range(ns):
    acc.add_shot(op.compute_shot_gradient(s, model, observed[s]))
result2 = acc.finalize()
```

`result.shots` 淇濆瓨閫愮偖鐨勫師濮嬫搴︺€乵isfit銆佸悎鎴?娈嬪樊锛屽彲鐩存帴妫€鏌ュ摢涓€鐐?璐＄尞寮傚父銆?
---

## 7. 绂绘暎涓€鑷存€ц鐐?
1. **杞疆涓ユ牸**锛歚adjoint_step` 鏄?$F^T$锛屼笉鏄繛缁柟绋嬬殑杩戜技绂绘暎锛涚敤
   `tests/test_adjoint.py` 鐨勫唴绉亽绛夊紡楠岃瘉銆?2. **鏃堕棿椤哄簭**锛氭畫宸敞鍏ャ€佸垰搴︽搴︺€侀€熷害浼撮殢鎺ㄨ繘銆佸瘑搴︽搴︺€佸簲鍔涗即闅?   鎺ㄨ繘鐨勯『搴忎笉鑳戒氦鎹紱瀵嗗害蹇呴』鐢ㄦ帹杩涘悗鐨勯€熷害浼撮殢銆?3. **闃诲凹**锛氭婕旀湯灏句箻 $G$锛屽弽鍚戝紑濮嬩箻 $G$銆?4. **骞冲潎绠楀瓙**锛?C_{44}$ 瑙掔偣骞冲潎銆?\rho_x/\rho_z$ 绠楁湳骞冲潎鐨勫弽鍚戝垎閰?   涓庢婕斾竴鑷达紱`np.pad(mode="edge")` 浜х敓鐨勯噸澶嶈妭鐐瑰湪杈圭晫鎶樺彔銆?5. **瀛樺偍鎶界█**锛氬唴瀛樹笉瓒虫椂 `ForwardStorage.stride>1`锛屾搴︾敤鏈€杩戝瓨鍌ㄦ椂鍒?   鐨勫満杩戜技锛沗stride=1` 鏃舵槸涓ユ牸绂绘暎姊害銆傛祴璇曞潎鐢?`stride=1`銆?
---

## 8. 鏈夐檺宸垎鏂瑰悜瀵兼暟妫€楠?
### 8.1 鏂规硶

閫夋柟鍚?$d$锛屼腑蹇冨樊鍒?
$$
D_{FD}=\frac{J(m+\epsilon d)-J(m-\epsilon d)}{2\epsilon},
$$

涓庝即闅忔柟鍚戝鏁?
$$
D_{ADJ}=\sum_p g_p\,d_p
$$

姣旇緝鐩稿璇樊銆傛楠岃剼鏈細`scripts/check_gradient.py`锛涘崟鍏冩祴璇曪細
`tests/test_gradient.py::test_directional_derivative_matches_fd`銆?
### 8.2 褰撳墠娴嬭瘯缁撴灉

**鍗曞厓娴嬭瘯**锛?6脳30銆? 鐐€?60 姝ワ級锛?
| 鏂瑰悜 | 鍙傛暟 | 浼撮殢 | 鏈夐檺宸垎 | 鐩稿璇樊 |
|---|---|---:|---:|---:|
| gaussian | vp | -5.312574e-09 | -5.312430e-09 | 2.72e-05 |
| gaussian | vs | -9.141950e-09 | -9.141787e-09 | 1.79e-05 |
| gaussian | rho | 2.137459e-11 | 2.136891e-11 | 2.66e-04 |
| random_smooth | vp | -2.137544e-09 | -2.137610e-09 | 3.07e-05 |
| random_smooth | vs | -1.820402e-09 | -1.820393e-09 | 4.93e-06 |
| random_smooth | rho | -7.983630e-10 | -7.983629e-10 | 8.51e-08 |

**婕旂ず鑴氭湰**锛?1脳61銆? 鐐€?58 姝ワ紝`scripts/check_gradient.py`锛夛細

| 鏂瑰悜 | 鍙傛暟 | 浼撮殢 | 鏈夐檺宸垎 | 鐩稿璇樊 |
|---|---|---:|---:|---:|
| gaussian | vp | 4.642620e-04 | 4.642592e-04 | 5.84e-06 |
| gaussian | vs | 2.175769e-05 | 2.175763e-05 | 2.68e-06 |
| gaussian | rho | -1.996298e-06 | -1.996299e-06 | 2.31e-07 |
| random_smooth | vp | 7.764439e-05 | 7.764429e-05 | 1.26e-06 |
| random_smooth | vs | -2.123332e-05 | -2.123332e-05 | 8.11e-09 |
| random_smooth | rho | -1.861060e-06 | -1.861062e-06 | 9.32e-07 |

rho 鐨勯珮鏂柟鍚戝湪鍗曞厓娴嬭瘯涓宸暐澶э紝鏄洜涓哄瘑搴︽€绘搴︿腑**璐ㄩ噺椤逛笌鍒氬害
閾惧紡椤规帴杩戞姷娑?*锛堜袱椤圭害 $7\times10^{-9}$锛屾€绘搴︾害 $2\times10^{-11}$锛夛紝
鐩稿璇樊琚斁澶э紱缁濆璇樊绾?$4\times10^{-13}$銆傛紨绀鸿剼鏈腑鏂瑰悜/灏哄害涓嶅悓锛?鐩稿璇樊涓?$2.3\times10^{-7}$銆傞€愰」妫€楠岋細

* 浠呰川閲忛」锛堝浐瀹?C锛屽彧鏀?$\rho$锛夛細鏂瑰悜瀵兼暟璇樊 $<10^{-6}$锛?* 浠呭垰搴﹂摼寮忛」锛堝浐瀹氳川閲忥紝鍙敼 C锛夛細璇樊 $4.5\times10^{-8}$銆?
### 8.3 瀵嗗害鏃堕棿椤哄簭鍥炲綊

鍚屼竴闅忔満瑙傛祴銆佸悓妯″瀷锛屽彧鏀硅褰曢暱搴︼紝姣旇緝瀵嗗害鏂瑰悜瀵兼暟锛?
| nt | 淇鍓?adj/fd | 淇鍚?adj/fd |
|---:|---:|---:|
| 3 | 0.9982 | 0.9999998 |
| 5 | 1.0474 | 1.0000001 |
| 10 | 1.0632 | 1.0000001 |
| 30 | 1.0608 | 1.0000001 |

璇存槑瀵嗗害椤瑰繀椤荤敤 $v虄^{n-1/2}$ 鐨勯『搴忎慨姝ｃ€?
### 8.4 杩愯

```powershell
.venv\Scripts\python.exe scripts\check_gradient.py
.venv\Scripts\python.exe -m unittest tests.test_gradient -v
```

褰撳墠瀹屾暣娴嬭瘯闆?**46 涓祴璇曞叏閮ㄩ€氳繃**锛堝惈鍚勫悜寮傛€?epsilon/delta 姊害銆?鍗曠偖/澶氱偖绱Н銆佷即闅忚浆缃亽绛夊紡锛夈€?
---

## 9. 鎺ュ彛璇存槑

| 鎺ュ彛 | 浣滅敤 |
|---|---|
| `efwi.adjoint.data_residual(syn, obs)` | 娈嬪樊涓庤娴嬭兘閲?|
| `efwi.adjoint.objective(res)` | 褰掍竴鍖?鍘熷鐩爣鍑芥暟 |
| `efwi.adjoint.adjoint_source(res, gain=1)` | 鏁版嵁绌洪棿浼撮殢婧?|
| `Elastic2DOperator.compute_shot_gradient(s, model, obs_s)` | 鍗曠偖鍘熷姊害 |
| `Elastic2DOperator.compute_gradient(model, obs, keep_shot_gradients=False)` | 鍏ㄩ儴鐐綊涓€鍖栨搴︼紙榛樿涓嶄繚鐣欓€愮偖鏁版嵁锛?|
| `efwi.adjoint.ShotGradientAccumulator` | 閫愮偖绱Н + 褰掍竴鍖?|
| `efwi.adjoint.compute_adjoint_gradients(op, model, obs)` | 楂樺眰 API锛岃繑鍥?vp/vs/rho |
| `efwi.adjoint.directional_derivative(...)` | 涓績宸垎鏂瑰悜瀵兼暟 |
| `efwi.adjoint.check_gradient(...)` | 閫愬弬鏁版湁闄愬樊鍒嗘楠?|
| `efwi.gradient.convert_gradient(...)` | 鍘熺敓 C/rho -> vp/vs/rho |

`ShotGradient` 瀛楁锛?
| 瀛楁 | 鍚箟 |
|---|---|
| `misfit_raw` | $0.5\sum r^2$锛堟湭褰掍竴鍖栵級 |
| `g` | `c11/c13/c33/c44/rho` 鍘熺敓姊害锛堢墿鐞嗙綉鏍硷級 |
| `h` | 浼?Hessian 瀵硅 |
| `synthetic/residuals` | 鍗曠偖鍚堟垚鏁版嵁涓庢畫宸?|
| `storage_stride` | 姝ｆ紨瀛樺偍鎶界█姝ラ暱 |

---

## 10. 澶嶇幇鍛戒护

```powershell
# 姊害鏂瑰悜瀵兼暟妫€楠岋紙鎵撳嵃琛ㄦ牸骞跺啓 JSON锛?.venv\Scripts\python.exe scripts\check_gradient.py

# 鍏ㄩ儴娴嬭瘯
.venv\Scripts\python.exe -m unittest discover -s tests -v

# 瀹屾暣 EFWI + RTM 婕旂ず
.venv\Scripts\python.exe scripts\run_demo.py --iterations 3
```

