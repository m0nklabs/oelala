(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))a(s);new MutationObserver(s=>{for(const l of s)if(l.type==="childList")for(const o of l.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&a(o)}).observe(document,{childList:!0,subtree:!0});function n(s){const l={};return s.integrity&&(l.integrity=s.integrity),s.referrerPolicy&&(l.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?l.credentials="include":s.crossOrigin==="anonymous"?l.credentials="omit":l.credentials="same-origin",l}function a(s){if(s.ep)return;s.ep=!0;const l=n(s);fetch(s.href,l)}})();function kp(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}var Qc={exports:{}},_s={},qc={exports:{}},he={};/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var ma=Symbol.for("react.element"),Sp=Symbol.for("react.portal"),Np=Symbol.for("react.fragment"),Cp=Symbol.for("react.strict_mode"),_p=Symbol.for("react.profiler"),zp=Symbol.for("react.provider"),Ep=Symbol.for("react.context"),Tp=Symbol.for("react.forward_ref"),Pp=Symbol.for("react.suspense"),Ip=Symbol.for("react.memo"),Rp=Symbol.for("react.lazy"),wi=Symbol.iterator;function Mp(e){return e===null||typeof e!="object"?null:(e=wi&&e[wi]||e["@@iterator"],typeof e=="function"?e:null)}var Yc={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},Xc=Object.assign,Kc={};function xn(e,t,n){this.props=e,this.context=t,this.refs=Kc,this.updater=n||Yc}xn.prototype.isReactComponent={};xn.prototype.setState=function(e,t){if(typeof e!="object"&&typeof e!="function"&&e!=null)throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,e,t,"setState")};xn.prototype.forceUpdate=function(e){this.updater.enqueueForceUpdate(this,e,"forceUpdate")};function Jc(){}Jc.prototype=xn.prototype;function wo(e,t,n){this.props=e,this.context=t,this.refs=Kc,this.updater=n||Yc}var ko=wo.prototype=new Jc;ko.constructor=wo;Xc(ko,xn.prototype);ko.isPureReactComponent=!0;var ki=Array.isArray,Zc=Object.prototype.hasOwnProperty,So={current:null},ed={key:!0,ref:!0,__self:!0,__source:!0};function td(e,t,n){var a,s={},l=null,o=null;if(t!=null)for(a in t.ref!==void 0&&(o=t.ref),t.key!==void 0&&(l=""+t.key),t)Zc.call(t,a)&&!ed.hasOwnProperty(a)&&(s[a]=t[a]);var c=arguments.length-2;if(c===1)s.children=n;else if(1<c){for(var d=Array(c),p=0;p<c;p++)d[p]=arguments[p+2];s.children=d}if(e&&e.defaultProps)for(a in c=e.defaultProps,c)s[a]===void 0&&(s[a]=c[a]);return{$$typeof:ma,type:e,key:l,ref:o,props:s,_owner:So.current}}function Fp(e,t){return{$$typeof:ma,type:e.type,key:t,ref:e.ref,props:e.props,_owner:e._owner}}function No(e){return typeof e=="object"&&e!==null&&e.$$typeof===ma}function Lp(e){var t={"=":"=0",":":"=2"};return"$"+e.replace(/[=:]/g,function(n){return t[n]})}var Si=/\/+/g;function Qs(e,t){return typeof e=="object"&&e!==null&&e.key!=null?Lp(""+e.key):t.toString(36)}function Ba(e,t,n,a,s){var l=typeof e;(l==="undefined"||l==="boolean")&&(e=null);var o=!1;if(e===null)o=!0;else switch(l){case"string":case"number":o=!0;break;case"object":switch(e.$$typeof){case ma:case Sp:o=!0}}if(o)return o=e,s=s(o),e=a===""?"."+Qs(o,0):a,ki(s)?(n="",e!=null&&(n=e.replace(Si,"$&/")+"/"),Ba(s,t,n,"",function(p){return p})):s!=null&&(No(s)&&(s=Fp(s,n+(!s.key||o&&o.key===s.key?"":(""+s.key).replace(Si,"$&/")+"/")+e)),t.push(s)),1;if(o=0,a=a===""?".":a+":",ki(e))for(var c=0;c<e.length;c++){l=e[c];var d=a+Qs(l,c);o+=Ba(l,t,n,d,s)}else if(d=Mp(e),typeof d=="function")for(e=d.call(e),c=0;!(l=e.next()).done;)l=l.value,d=a+Qs(l,c++),o+=Ba(l,t,n,d,s);else if(l==="object")throw t=String(e),Error("Objects are not valid as a React child (found: "+(t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t)+"). If you meant to render a collection of children, use an array instead.");return o}function Na(e,t,n){if(e==null)return e;var a=[],s=0;return Ba(e,a,"","",function(l){return t.call(n,l,s++)}),a}function Dp(e){if(e._status===-1){var t=e._result;t=t(),t.then(function(n){(e._status===0||e._status===-1)&&(e._status=1,e._result=n)},function(n){(e._status===0||e._status===-1)&&(e._status=2,e._result=n)}),e._status===-1&&(e._status=0,e._result=t)}if(e._status===1)return e._result.default;throw e._result}var Xe={current:null},Wa={transition:null},Op={ReactCurrentDispatcher:Xe,ReactCurrentBatchConfig:Wa,ReactCurrentOwner:So};function rd(){throw Error("act(...) is not supported in production builds of React.")}he.Children={map:Na,forEach:function(e,t,n){Na(e,function(){t.apply(this,arguments)},n)},count:function(e){var t=0;return Na(e,function(){t++}),t},toArray:function(e){return Na(e,function(t){return t})||[]},only:function(e){if(!No(e))throw Error("React.Children.only expected to receive a single React element child.");return e}};he.Component=xn;he.Fragment=Np;he.Profiler=_p;he.PureComponent=wo;he.StrictMode=Cp;he.Suspense=Pp;he.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=Op;he.act=rd;he.cloneElement=function(e,t,n){if(e==null)throw Error("React.cloneElement(...): The argument must be a React element, but you passed "+e+".");var a=Xc({},e.props),s=e.key,l=e.ref,o=e._owner;if(t!=null){if(t.ref!==void 0&&(l=t.ref,o=So.current),t.key!==void 0&&(s=""+t.key),e.type&&e.type.defaultProps)var c=e.type.defaultProps;for(d in t)Zc.call(t,d)&&!ed.hasOwnProperty(d)&&(a[d]=t[d]===void 0&&c!==void 0?c[d]:t[d])}var d=arguments.length-2;if(d===1)a.children=n;else if(1<d){c=Array(d);for(var p=0;p<d;p++)c[p]=arguments[p+2];a.children=c}return{$$typeof:ma,type:e.type,key:s,ref:l,props:a,_owner:o}};he.createContext=function(e){return e={$$typeof:Ep,_currentValue:e,_currentValue2:e,_threadCount:0,Provider:null,Consumer:null,_defaultValue:null,_globalName:null},e.Provider={$$typeof:zp,_context:e},e.Consumer=e};he.createElement=td;he.createFactory=function(e){var t=td.bind(null,e);return t.type=e,t};he.createRef=function(){return{current:null}};he.forwardRef=function(e){return{$$typeof:Tp,render:e}};he.isValidElement=No;he.lazy=function(e){return{$$typeof:Rp,_payload:{_status:-1,_result:e},_init:Dp}};he.memo=function(e,t){return{$$typeof:Ip,type:e,compare:t===void 0?null:t}};he.startTransition=function(e){var t=Wa.transition;Wa.transition={};try{e()}finally{Wa.transition=t}};he.unstable_act=rd;he.useCallback=function(e,t){return Xe.current.useCallback(e,t)};he.useContext=function(e){return Xe.current.useContext(e)};he.useDebugValue=function(){};he.useDeferredValue=function(e){return Xe.current.useDeferredValue(e)};he.useEffect=function(e,t){return Xe.current.useEffect(e,t)};he.useId=function(){return Xe.current.useId()};he.useImperativeHandle=function(e,t,n){return Xe.current.useImperativeHandle(e,t,n)};he.useInsertionEffect=function(e,t){return Xe.current.useInsertionEffect(e,t)};he.useLayoutEffect=function(e,t){return Xe.current.useLayoutEffect(e,t)};he.useMemo=function(e,t){return Xe.current.useMemo(e,t)};he.useReducer=function(e,t,n){return Xe.current.useReducer(e,t,n)};he.useRef=function(e){return Xe.current.useRef(e)};he.useState=function(e){return Xe.current.useState(e)};he.useSyncExternalStore=function(e,t,n){return Xe.current.useSyncExternalStore(e,t,n)};he.useTransition=function(){return Xe.current.useTransition()};he.version="18.3.1";qc.exports=he;var i=qc.exports;const Ap=kp(i);/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var $p=i,Up=Symbol.for("react.element"),Vp=Symbol.for("react.fragment"),Bp=Object.prototype.hasOwnProperty,Wp=$p.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,Hp={key:!0,ref:!0,__self:!0,__source:!0};function nd(e,t,n){var a,s={},l=null,o=null;n!==void 0&&(l=""+n),t.key!==void 0&&(l=""+t.key),t.ref!==void 0&&(o=t.ref);for(a in t)Bp.call(t,a)&&!Hp.hasOwnProperty(a)&&(s[a]=t[a]);if(e&&e.defaultProps)for(a in t=e.defaultProps,t)s[a]===void 0&&(s[a]=t[a]);return{$$typeof:Up,type:e,key:l,ref:o,props:s,_owner:Wp.current}}_s.Fragment=Vp;_s.jsx=nd;_s.jsxs=nd;Qc.exports=_s;var r=Qc.exports,Sl={},ad={exports:{}},ut={},sd={exports:{}},ld={};/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */(function(e){function t(D,U){var q=D.length;D.push(U);e:for(;0<q;){var V=q-1>>>1,H=D[V];if(0<s(H,U))D[V]=U,D[q]=H,q=V;else break e}}function n(D){return D.length===0?null:D[0]}function a(D){if(D.length===0)return null;var U=D[0],q=D.pop();if(q!==U){D[0]=q;e:for(var V=0,H=D.length,Q=H>>>1;V<Q;){var C=2*(V+1)-1,Y=D[C],M=C+1,m=D[M];if(0>s(Y,q))M<H&&0>s(m,Y)?(D[V]=m,D[M]=q,V=M):(D[V]=Y,D[C]=q,V=C);else if(M<H&&0>s(m,q))D[V]=m,D[M]=q,V=M;else break e}}return U}function s(D,U){var q=D.sortIndex-U.sortIndex;return q!==0?q:D.id-U.id}if(typeof performance=="object"&&typeof performance.now=="function"){var l=performance;e.unstable_now=function(){return l.now()}}else{var o=Date,c=o.now();e.unstable_now=function(){return o.now()-c}}var d=[],p=[],v=1,g=null,x=3,k=!1,w=!1,z=!1,F=typeof setTimeout=="function"?setTimeout:null,f=typeof clearTimeout=="function"?clearTimeout:null,u=typeof setImmediate<"u"?setImmediate:null;typeof navigator<"u"&&navigator.scheduling!==void 0&&navigator.scheduling.isInputPending!==void 0&&navigator.scheduling.isInputPending.bind(navigator.scheduling);function h(D){for(var U=n(p);U!==null;){if(U.callback===null)a(p);else if(U.startTime<=D)a(p),U.sortIndex=U.expirationTime,t(d,U);else break;U=n(p)}}function y(D){if(z=!1,h(D),!w)if(n(d)!==null)w=!0,ne(j);else{var U=n(p);U!==null&&ae(y,U.startTime-D)}}function j(D,U){w=!1,z&&(z=!1,f(R),R=-1),k=!0;var q=x;try{for(h(U),g=n(d);g!==null&&(!(g.expirationTime>U)||D&&!b());){var V=g.callback;if(typeof V=="function"){g.callback=null,x=g.priorityLevel;var H=V(g.expirationTime<=U);U=e.unstable_now(),typeof H=="function"?g.callback=H:g===n(d)&&a(d),h(U)}else a(d);g=n(d)}if(g!==null)var Q=!0;else{var C=n(p);C!==null&&ae(y,C.startTime-U),Q=!1}return Q}finally{g=null,x=q,k=!1}}var I=!1,_=null,R=-1,G=5,W=-1;function b(){return!(e.unstable_now()-W<G)}function N(){if(_!==null){var D=e.unstable_now();W=D;var U=!0;try{U=_(!0,D)}finally{U?L():(I=!1,_=null)}}else I=!1}var L;if(typeof u=="function")L=function(){u(N)};else if(typeof MessageChannel<"u"){var ee=new MessageChannel,T=ee.port2;ee.port1.onmessage=N,L=function(){T.postMessage(null)}}else L=function(){F(N,0)};function ne(D){_=D,I||(I=!0,L())}function ae(D,U){R=F(function(){D(e.unstable_now())},U)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(D){D.callback=null},e.unstable_continueExecution=function(){w||k||(w=!0,ne(j))},e.unstable_forceFrameRate=function(D){0>D||125<D?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):G=0<D?Math.floor(1e3/D):5},e.unstable_getCurrentPriorityLevel=function(){return x},e.unstable_getFirstCallbackNode=function(){return n(d)},e.unstable_next=function(D){switch(x){case 1:case 2:case 3:var U=3;break;default:U=x}var q=x;x=U;try{return D()}finally{x=q}},e.unstable_pauseExecution=function(){},e.unstable_requestPaint=function(){},e.unstable_runWithPriority=function(D,U){switch(D){case 1:case 2:case 3:case 4:case 5:break;default:D=3}var q=x;x=D;try{return U()}finally{x=q}},e.unstable_scheduleCallback=function(D,U,q){var V=e.unstable_now();switch(typeof q=="object"&&q!==null?(q=q.delay,q=typeof q=="number"&&0<q?V+q:V):q=V,D){case 1:var H=-1;break;case 2:H=250;break;case 5:H=1073741823;break;case 4:H=1e4;break;default:H=5e3}return H=q+H,D={id:v++,callback:U,priorityLevel:D,startTime:q,expirationTime:H,sortIndex:-1},q>V?(D.sortIndex=q,t(p,D),n(d)===null&&D===n(p)&&(z?(f(R),R=-1):z=!0,ae(y,q-V))):(D.sortIndex=H,t(d,D),w||k||(w=!0,ne(j))),D},e.unstable_shouldYield=b,e.unstable_wrapCallback=function(D){var U=x;return function(){var q=x;x=U;try{return D.apply(this,arguments)}finally{x=q}}}})(ld);sd.exports=ld;var Gp=sd.exports;/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Qp=i,dt=Gp;function B(e){for(var t="https://reactjs.org/docs/error-decoder.html?invariant="+e,n=1;n<arguments.length;n++)t+="&args[]="+encodeURIComponent(arguments[n]);return"Minified React error #"+e+"; visit "+t+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}var od=new Set,Qn={};function Dr(e,t){ln(e,t),ln(e+"Capture",t)}function ln(e,t){for(Qn[e]=t,e=0;e<t.length;e++)od.add(t[e])}var Qt=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),Nl=Object.prototype.hasOwnProperty,qp=/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,Ni={},Ci={};function Yp(e){return Nl.call(Ci,e)?!0:Nl.call(Ni,e)?!1:qp.test(e)?Ci[e]=!0:(Ni[e]=!0,!1)}function Xp(e,t,n,a){if(n!==null&&n.type===0)return!1;switch(typeof t){case"function":case"symbol":return!0;case"boolean":return a?!1:n!==null?!n.acceptsBooleans:(e=e.toLowerCase().slice(0,5),e!=="data-"&&e!=="aria-");default:return!1}}function Kp(e,t,n,a){if(t===null||typeof t>"u"||Xp(e,t,n,a))return!0;if(a)return!1;if(n!==null)switch(n.type){case 3:return!t;case 4:return t===!1;case 5:return isNaN(t);case 6:return isNaN(t)||1>t}return!1}function Ke(e,t,n,a,s,l,o){this.acceptsBooleans=t===2||t===3||t===4,this.attributeName=a,this.attributeNamespace=s,this.mustUseProperty=n,this.propertyName=e,this.type=t,this.sanitizeURL=l,this.removeEmptyString=o}var Ae={};"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e){Ae[e]=new Ke(e,0,!1,e,null,!1,!1)});[["acceptCharset","accept-charset"],["className","class"],["htmlFor","for"],["httpEquiv","http-equiv"]].forEach(function(e){var t=e[0];Ae[t]=new Ke(t,1,!1,e[1],null,!1,!1)});["contentEditable","draggable","spellCheck","value"].forEach(function(e){Ae[e]=new Ke(e,2,!1,e.toLowerCase(),null,!1,!1)});["autoReverse","externalResourcesRequired","focusable","preserveAlpha"].forEach(function(e){Ae[e]=new Ke(e,2,!1,e,null,!1,!1)});"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e){Ae[e]=new Ke(e,3,!1,e.toLowerCase(),null,!1,!1)});["checked","multiple","muted","selected"].forEach(function(e){Ae[e]=new Ke(e,3,!0,e,null,!1,!1)});["capture","download"].forEach(function(e){Ae[e]=new Ke(e,4,!1,e,null,!1,!1)});["cols","rows","size","span"].forEach(function(e){Ae[e]=new Ke(e,6,!1,e,null,!1,!1)});["rowSpan","start"].forEach(function(e){Ae[e]=new Ke(e,5,!1,e.toLowerCase(),null,!1,!1)});var Co=/[\-:]([a-z])/g;function _o(e){return e[1].toUpperCase()}"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e){var t=e.replace(Co,_o);Ae[t]=new Ke(t,1,!1,e,null,!1,!1)});"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e){var t=e.replace(Co,_o);Ae[t]=new Ke(t,1,!1,e,"http://www.w3.org/1999/xlink",!1,!1)});["xml:base","xml:lang","xml:space"].forEach(function(e){var t=e.replace(Co,_o);Ae[t]=new Ke(t,1,!1,e,"http://www.w3.org/XML/1998/namespace",!1,!1)});["tabIndex","crossOrigin"].forEach(function(e){Ae[e]=new Ke(e,1,!1,e.toLowerCase(),null,!1,!1)});Ae.xlinkHref=new Ke("xlinkHref",1,!1,"xlink:href","http://www.w3.org/1999/xlink",!0,!1);["src","href","action","formAction"].forEach(function(e){Ae[e]=new Ke(e,1,!1,e.toLowerCase(),null,!0,!0)});function zo(e,t,n,a){var s=Ae.hasOwnProperty(t)?Ae[t]:null;(s!==null?s.type!==0:a||!(2<t.length)||t[0]!=="o"&&t[0]!=="O"||t[1]!=="n"&&t[1]!=="N")&&(Kp(t,n,s,a)&&(n=null),a||s===null?Yp(t)&&(n===null?e.removeAttribute(t):e.setAttribute(t,""+n)):s.mustUseProperty?e[s.propertyName]=n===null?s.type===3?!1:"":n:(t=s.attributeName,a=s.attributeNamespace,n===null?e.removeAttribute(t):(s=s.type,n=s===3||s===4&&n===!0?"":""+n,a?e.setAttributeNS(a,t,n):e.setAttribute(t,n))))}var Jt=Qp.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,Ca=Symbol.for("react.element"),Vr=Symbol.for("react.portal"),Br=Symbol.for("react.fragment"),Eo=Symbol.for("react.strict_mode"),Cl=Symbol.for("react.profiler"),id=Symbol.for("react.provider"),cd=Symbol.for("react.context"),To=Symbol.for("react.forward_ref"),_l=Symbol.for("react.suspense"),zl=Symbol.for("react.suspense_list"),Po=Symbol.for("react.memo"),tr=Symbol.for("react.lazy"),dd=Symbol.for("react.offscreen"),_i=Symbol.iterator;function Cn(e){return e===null||typeof e!="object"?null:(e=_i&&e[_i]||e["@@iterator"],typeof e=="function"?e:null)}var _e=Object.assign,qs;function Mn(e){if(qs===void 0)try{throw Error()}catch(n){var t=n.stack.trim().match(/\n( *(at )?)/);qs=t&&t[1]||""}return`
`+qs+e}var Ys=!1;function Xs(e,t){if(!e||Ys)return"";Ys=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{if(t)if(t=function(){throw Error()},Object.defineProperty(t.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(t,[])}catch(p){var a=p}Reflect.construct(e,[],t)}else{try{t.call()}catch(p){a=p}e.call(t.prototype)}else{try{throw Error()}catch(p){a=p}e()}}catch(p){if(p&&a&&typeof p.stack=="string"){for(var s=p.stack.split(`
`),l=a.stack.split(`
`),o=s.length-1,c=l.length-1;1<=o&&0<=c&&s[o]!==l[c];)c--;for(;1<=o&&0<=c;o--,c--)if(s[o]!==l[c]){if(o!==1||c!==1)do if(o--,c--,0>c||s[o]!==l[c]){var d=`
`+s[o].replace(" at new "," at ");return e.displayName&&d.includes("<anonymous>")&&(d=d.replace("<anonymous>",e.displayName)),d}while(1<=o&&0<=c);break}}}finally{Ys=!1,Error.prepareStackTrace=n}return(e=e?e.displayName||e.name:"")?Mn(e):""}function Jp(e){switch(e.tag){case 5:return Mn(e.type);case 16:return Mn("Lazy");case 13:return Mn("Suspense");case 19:return Mn("SuspenseList");case 0:case 2:case 15:return e=Xs(e.type,!1),e;case 11:return e=Xs(e.type.render,!1),e;case 1:return e=Xs(e.type,!0),e;default:return""}}function El(e){if(e==null)return null;if(typeof e=="function")return e.displayName||e.name||null;if(typeof e=="string")return e;switch(e){case Br:return"Fragment";case Vr:return"Portal";case Cl:return"Profiler";case Eo:return"StrictMode";case _l:return"Suspense";case zl:return"SuspenseList"}if(typeof e=="object")switch(e.$$typeof){case cd:return(e.displayName||"Context")+".Consumer";case id:return(e._context.displayName||"Context")+".Provider";case To:var t=e.render;return e=e.displayName,e||(e=t.displayName||t.name||"",e=e!==""?"ForwardRef("+e+")":"ForwardRef"),e;case Po:return t=e.displayName||null,t!==null?t:El(e.type)||"Memo";case tr:t=e._payload,e=e._init;try{return El(e(t))}catch{}}return null}function Zp(e){var t=e.type;switch(e.tag){case 24:return"Cache";case 9:return(t.displayName||"Context")+".Consumer";case 10:return(t._context.displayName||"Context")+".Provider";case 18:return"DehydratedFragment";case 11:return e=t.render,e=e.displayName||e.name||"",t.displayName||(e!==""?"ForwardRef("+e+")":"ForwardRef");case 7:return"Fragment";case 5:return t;case 4:return"Portal";case 3:return"Root";case 6:return"Text";case 16:return El(t);case 8:return t===Eo?"StrictMode":"Mode";case 22:return"Offscreen";case 12:return"Profiler";case 21:return"Scope";case 13:return"Suspense";case 19:return"SuspenseList";case 25:return"TracingMarker";case 1:case 0:case 17:case 2:case 14:case 15:if(typeof t=="function")return t.displayName||t.name||null;if(typeof t=="string")return t}return null}function hr(e){switch(typeof e){case"boolean":case"number":case"string":case"undefined":return e;case"object":return e;default:return""}}function ud(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()==="input"&&(t==="checkbox"||t==="radio")}function ef(e){var t=ud(e)?"checked":"value",n=Object.getOwnPropertyDescriptor(e.constructor.prototype,t),a=""+e[t];if(!e.hasOwnProperty(t)&&typeof n<"u"&&typeof n.get=="function"&&typeof n.set=="function"){var s=n.get,l=n.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return s.call(this)},set:function(o){a=""+o,l.call(this,o)}}),Object.defineProperty(e,t,{enumerable:n.enumerable}),{getValue:function(){return a},setValue:function(o){a=""+o},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function _a(e){e._valueTracker||(e._valueTracker=ef(e))}function pd(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),a="";return e&&(a=ud(e)?e.checked?"true":"false":e.value),e=a,e!==n?(t.setValue(e),!0):!1}function ts(e){if(e=e||(typeof document<"u"?document:void 0),typeof e>"u")return null;try{return e.activeElement||e.body}catch{return e.body}}function Tl(e,t){var n=t.checked;return _e({},t,{defaultChecked:void 0,defaultValue:void 0,value:void 0,checked:n??e._wrapperState.initialChecked})}function zi(e,t){var n=t.defaultValue==null?"":t.defaultValue,a=t.checked!=null?t.checked:t.defaultChecked;n=hr(t.value!=null?t.value:n),e._wrapperState={initialChecked:a,initialValue:n,controlled:t.type==="checkbox"||t.type==="radio"?t.checked!=null:t.value!=null}}function fd(e,t){t=t.checked,t!=null&&zo(e,"checked",t,!1)}function Pl(e,t){fd(e,t);var n=hr(t.value),a=t.type;if(n!=null)a==="number"?(n===0&&e.value===""||e.value!=n)&&(e.value=""+n):e.value!==""+n&&(e.value=""+n);else if(a==="submit"||a==="reset"){e.removeAttribute("value");return}t.hasOwnProperty("value")?Il(e,t.type,n):t.hasOwnProperty("defaultValue")&&Il(e,t.type,hr(t.defaultValue)),t.checked==null&&t.defaultChecked!=null&&(e.defaultChecked=!!t.defaultChecked)}function Ei(e,t,n){if(t.hasOwnProperty("value")||t.hasOwnProperty("defaultValue")){var a=t.type;if(!(a!=="submit"&&a!=="reset"||t.value!==void 0&&t.value!==null))return;t=""+e._wrapperState.initialValue,n||t===e.value||(e.value=t),e.defaultValue=t}n=e.name,n!==""&&(e.name=""),e.defaultChecked=!!e._wrapperState.initialChecked,n!==""&&(e.name=n)}function Il(e,t,n){(t!=="number"||ts(e.ownerDocument)!==e)&&(n==null?e.defaultValue=""+e._wrapperState.initialValue:e.defaultValue!==""+n&&(e.defaultValue=""+n))}var Fn=Array.isArray;function en(e,t,n,a){if(e=e.options,t){t={};for(var s=0;s<n.length;s++)t["$"+n[s]]=!0;for(n=0;n<e.length;n++)s=t.hasOwnProperty("$"+e[n].value),e[n].selected!==s&&(e[n].selected=s),s&&a&&(e[n].defaultSelected=!0)}else{for(n=""+hr(n),t=null,s=0;s<e.length;s++){if(e[s].value===n){e[s].selected=!0,a&&(e[s].defaultSelected=!0);return}t!==null||e[s].disabled||(t=e[s])}t!==null&&(t.selected=!0)}}function Rl(e,t){if(t.dangerouslySetInnerHTML!=null)throw Error(B(91));return _e({},t,{value:void 0,defaultValue:void 0,children:""+e._wrapperState.initialValue})}function Ti(e,t){var n=t.value;if(n==null){if(n=t.children,t=t.defaultValue,n!=null){if(t!=null)throw Error(B(92));if(Fn(n)){if(1<n.length)throw Error(B(93));n=n[0]}t=n}t==null&&(t=""),n=t}e._wrapperState={initialValue:hr(n)}}function md(e,t){var n=hr(t.value),a=hr(t.defaultValue);n!=null&&(n=""+n,n!==e.value&&(e.value=n),t.defaultValue==null&&e.defaultValue!==n&&(e.defaultValue=n)),a!=null&&(e.defaultValue=""+a)}function Pi(e){var t=e.textContent;t===e._wrapperState.initialValue&&t!==""&&t!==null&&(e.value=t)}function hd(e){switch(e){case"svg":return"http://www.w3.org/2000/svg";case"math":return"http://www.w3.org/1998/Math/MathML";default:return"http://www.w3.org/1999/xhtml"}}function Ml(e,t){return e==null||e==="http://www.w3.org/1999/xhtml"?hd(t):e==="http://www.w3.org/2000/svg"&&t==="foreignObject"?"http://www.w3.org/1999/xhtml":e}var za,xd=function(e){return typeof MSApp<"u"&&MSApp.execUnsafeLocalFunction?function(t,n,a,s){MSApp.execUnsafeLocalFunction(function(){return e(t,n,a,s)})}:e}(function(e,t){if(e.namespaceURI!=="http://www.w3.org/2000/svg"||"innerHTML"in e)e.innerHTML=t;else{for(za=za||document.createElement("div"),za.innerHTML="<svg>"+t.valueOf().toString()+"</svg>",t=za.firstChild;e.firstChild;)e.removeChild(e.firstChild);for(;t.firstChild;)e.appendChild(t.firstChild)}});function qn(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var On={animationIterationCount:!0,aspectRatio:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridArea:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},tf=["Webkit","ms","Moz","O"];Object.keys(On).forEach(function(e){tf.forEach(function(t){t=t+e.charAt(0).toUpperCase()+e.substring(1),On[t]=On[e]})});function gd(e,t,n){return t==null||typeof t=="boolean"||t===""?"":n||typeof t!="number"||t===0||On.hasOwnProperty(e)&&On[e]?(""+t).trim():t+"px"}function vd(e,t){e=e.style;for(var n in t)if(t.hasOwnProperty(n)){var a=n.indexOf("--")===0,s=gd(n,t[n],a);n==="float"&&(n="cssFloat"),a?e.setProperty(n,s):e[n]=s}}var rf=_e({menuitem:!0},{area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0});function Fl(e,t){if(t){if(rf[e]&&(t.children!=null||t.dangerouslySetInnerHTML!=null))throw Error(B(137,e));if(t.dangerouslySetInnerHTML!=null){if(t.children!=null)throw Error(B(60));if(typeof t.dangerouslySetInnerHTML!="object"||!("__html"in t.dangerouslySetInnerHTML))throw Error(B(61))}if(t.style!=null&&typeof t.style!="object")throw Error(B(62))}}function Ll(e,t){if(e.indexOf("-")===-1)return typeof t.is=="string";switch(e){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var Dl=null;function Io(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var Ol=null,tn=null,rn=null;function Ii(e){if(e=ga(e)){if(typeof Ol!="function")throw Error(B(280));var t=e.stateNode;t&&(t=Is(t),Ol(e.stateNode,e.type,t))}}function yd(e){tn?rn?rn.push(e):rn=[e]:tn=e}function jd(){if(tn){var e=tn,t=rn;if(rn=tn=null,Ii(e),t)for(e=0;e<t.length;e++)Ii(t[e])}}function bd(e,t){return e(t)}function wd(){}var Ks=!1;function kd(e,t,n){if(Ks)return e(t,n);Ks=!0;try{return bd(e,t,n)}finally{Ks=!1,(tn!==null||rn!==null)&&(wd(),jd())}}function Yn(e,t){var n=e.stateNode;if(n===null)return null;var a=Is(n);if(a===null)return null;n=a[t];e:switch(t){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(a=!a.disabled)||(e=e.type,a=!(e==="button"||e==="input"||e==="select"||e==="textarea")),e=!a;break e;default:e=!1}if(e)return null;if(n&&typeof n!="function")throw Error(B(231,t,typeof n));return n}var Al=!1;if(Qt)try{var _n={};Object.defineProperty(_n,"passive",{get:function(){Al=!0}}),window.addEventListener("test",_n,_n),window.removeEventListener("test",_n,_n)}catch{Al=!1}function nf(e,t,n,a,s,l,o,c,d){var p=Array.prototype.slice.call(arguments,3);try{t.apply(n,p)}catch(v){this.onError(v)}}var An=!1,rs=null,ns=!1,$l=null,af={onError:function(e){An=!0,rs=e}};function sf(e,t,n,a,s,l,o,c,d){An=!1,rs=null,nf.apply(af,arguments)}function lf(e,t,n,a,s,l,o,c,d){if(sf.apply(this,arguments),An){if(An){var p=rs;An=!1,rs=null}else throw Error(B(198));ns||(ns=!0,$l=p)}}function Or(e){var t=e,n=e;if(e.alternate)for(;t.return;)t=t.return;else{e=t;do t=e,t.flags&4098&&(n=t.return),e=t.return;while(e)}return t.tag===3?n:null}function Sd(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function Ri(e){if(Or(e)!==e)throw Error(B(188))}function of(e){var t=e.alternate;if(!t){if(t=Or(e),t===null)throw Error(B(188));return t!==e?null:e}for(var n=e,a=t;;){var s=n.return;if(s===null)break;var l=s.alternate;if(l===null){if(a=s.return,a!==null){n=a;continue}break}if(s.child===l.child){for(l=s.child;l;){if(l===n)return Ri(s),e;if(l===a)return Ri(s),t;l=l.sibling}throw Error(B(188))}if(n.return!==a.return)n=s,a=l;else{for(var o=!1,c=s.child;c;){if(c===n){o=!0,n=s,a=l;break}if(c===a){o=!0,a=s,n=l;break}c=c.sibling}if(!o){for(c=l.child;c;){if(c===n){o=!0,n=l,a=s;break}if(c===a){o=!0,a=l,n=s;break}c=c.sibling}if(!o)throw Error(B(189))}}if(n.alternate!==a)throw Error(B(190))}if(n.tag!==3)throw Error(B(188));return n.stateNode.current===n?e:t}function Nd(e){return e=of(e),e!==null?Cd(e):null}function Cd(e){if(e.tag===5||e.tag===6)return e;for(e=e.child;e!==null;){var t=Cd(e);if(t!==null)return t;e=e.sibling}return null}var _d=dt.unstable_scheduleCallback,Mi=dt.unstable_cancelCallback,cf=dt.unstable_shouldYield,df=dt.unstable_requestPaint,Ee=dt.unstable_now,uf=dt.unstable_getCurrentPriorityLevel,Ro=dt.unstable_ImmediatePriority,zd=dt.unstable_UserBlockingPriority,as=dt.unstable_NormalPriority,pf=dt.unstable_LowPriority,Ed=dt.unstable_IdlePriority,zs=null,Dt=null;function ff(e){if(Dt&&typeof Dt.onCommitFiberRoot=="function")try{Dt.onCommitFiberRoot(zs,e,void 0,(e.current.flags&128)===128)}catch{}}var _t=Math.clz32?Math.clz32:xf,mf=Math.log,hf=Math.LN2;function xf(e){return e>>>=0,e===0?32:31-(mf(e)/hf|0)|0}var Ea=64,Ta=4194304;function Ln(e){switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e&4194240;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return e&130023424;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 1073741824;default:return e}}function ss(e,t){var n=e.pendingLanes;if(n===0)return 0;var a=0,s=e.suspendedLanes,l=e.pingedLanes,o=n&268435455;if(o!==0){var c=o&~s;c!==0?a=Ln(c):(l&=o,l!==0&&(a=Ln(l)))}else o=n&~s,o!==0?a=Ln(o):l!==0&&(a=Ln(l));if(a===0)return 0;if(t!==0&&t!==a&&!(t&s)&&(s=a&-a,l=t&-t,s>=l||s===16&&(l&4194240)!==0))return t;if(a&4&&(a|=n&16),t=e.entangledLanes,t!==0)for(e=e.entanglements,t&=a;0<t;)n=31-_t(t),s=1<<n,a|=e[n],t&=~s;return a}function gf(e,t){switch(e){case 1:case 2:case 4:return t+250;case 8:case 16:case 32:case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return-1;case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function vf(e,t){for(var n=e.suspendedLanes,a=e.pingedLanes,s=e.expirationTimes,l=e.pendingLanes;0<l;){var o=31-_t(l),c=1<<o,d=s[o];d===-1?(!(c&n)||c&a)&&(s[o]=gf(c,t)):d<=t&&(e.expiredLanes|=c),l&=~c}}function Ul(e){return e=e.pendingLanes&-1073741825,e!==0?e:e&1073741824?1073741824:0}function Td(){var e=Ea;return Ea<<=1,!(Ea&4194240)&&(Ea=64),e}function Js(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function ha(e,t,n){e.pendingLanes|=t,t!==536870912&&(e.suspendedLanes=0,e.pingedLanes=0),e=e.eventTimes,t=31-_t(t),e[t]=n}function yf(e,t){var n=e.pendingLanes&~t;e.pendingLanes=t,e.suspendedLanes=0,e.pingedLanes=0,e.expiredLanes&=t,e.mutableReadLanes&=t,e.entangledLanes&=t,t=e.entanglements;var a=e.eventTimes;for(e=e.expirationTimes;0<n;){var s=31-_t(n),l=1<<s;t[s]=0,a[s]=-1,e[s]=-1,n&=~l}}function Mo(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var a=31-_t(n),s=1<<a;s&t|e[a]&t&&(e[a]|=t),n&=~s}}var je=0;function Pd(e){return e&=-e,1<e?4<e?e&268435455?16:536870912:4:1}var Id,Fo,Rd,Md,Fd,Vl=!1,Pa=[],or=null,ir=null,cr=null,Xn=new Map,Kn=new Map,nr=[],jf="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");function Fi(e,t){switch(e){case"focusin":case"focusout":or=null;break;case"dragenter":case"dragleave":ir=null;break;case"mouseover":case"mouseout":cr=null;break;case"pointerover":case"pointerout":Xn.delete(t.pointerId);break;case"gotpointercapture":case"lostpointercapture":Kn.delete(t.pointerId)}}function zn(e,t,n,a,s,l){return e===null||e.nativeEvent!==l?(e={blockedOn:t,domEventName:n,eventSystemFlags:a,nativeEvent:l,targetContainers:[s]},t!==null&&(t=ga(t),t!==null&&Fo(t)),e):(e.eventSystemFlags|=a,t=e.targetContainers,s!==null&&t.indexOf(s)===-1&&t.push(s),e)}function bf(e,t,n,a,s){switch(t){case"focusin":return or=zn(or,e,t,n,a,s),!0;case"dragenter":return ir=zn(ir,e,t,n,a,s),!0;case"mouseover":return cr=zn(cr,e,t,n,a,s),!0;case"pointerover":var l=s.pointerId;return Xn.set(l,zn(Xn.get(l)||null,e,t,n,a,s)),!0;case"gotpointercapture":return l=s.pointerId,Kn.set(l,zn(Kn.get(l)||null,e,t,n,a,s)),!0}return!1}function Ld(e){var t=_r(e.target);if(t!==null){var n=Or(t);if(n!==null){if(t=n.tag,t===13){if(t=Sd(n),t!==null){e.blockedOn=t,Fd(e.priority,function(){Rd(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function Ha(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=Bl(e.domEventName,e.eventSystemFlags,t[0],e.nativeEvent);if(n===null){n=e.nativeEvent;var a=new n.constructor(n.type,n);Dl=a,n.target.dispatchEvent(a),Dl=null}else return t=ga(n),t!==null&&Fo(t),e.blockedOn=n,!1;t.shift()}return!0}function Li(e,t,n){Ha(e)&&n.delete(t)}function wf(){Vl=!1,or!==null&&Ha(or)&&(or=null),ir!==null&&Ha(ir)&&(ir=null),cr!==null&&Ha(cr)&&(cr=null),Xn.forEach(Li),Kn.forEach(Li)}function En(e,t){e.blockedOn===t&&(e.blockedOn=null,Vl||(Vl=!0,dt.unstable_scheduleCallback(dt.unstable_NormalPriority,wf)))}function Jn(e){function t(s){return En(s,e)}if(0<Pa.length){En(Pa[0],e);for(var n=1;n<Pa.length;n++){var a=Pa[n];a.blockedOn===e&&(a.blockedOn=null)}}for(or!==null&&En(or,e),ir!==null&&En(ir,e),cr!==null&&En(cr,e),Xn.forEach(t),Kn.forEach(t),n=0;n<nr.length;n++)a=nr[n],a.blockedOn===e&&(a.blockedOn=null);for(;0<nr.length&&(n=nr[0],n.blockedOn===null);)Ld(n),n.blockedOn===null&&nr.shift()}var nn=Jt.ReactCurrentBatchConfig,ls=!0;function kf(e,t,n,a){var s=je,l=nn.transition;nn.transition=null;try{je=1,Lo(e,t,n,a)}finally{je=s,nn.transition=l}}function Sf(e,t,n,a){var s=je,l=nn.transition;nn.transition=null;try{je=4,Lo(e,t,n,a)}finally{je=s,nn.transition=l}}function Lo(e,t,n,a){if(ls){var s=Bl(e,t,n,a);if(s===null)il(e,t,a,os,n),Fi(e,a);else if(bf(s,e,t,n,a))a.stopPropagation();else if(Fi(e,a),t&4&&-1<jf.indexOf(e)){for(;s!==null;){var l=ga(s);if(l!==null&&Id(l),l=Bl(e,t,n,a),l===null&&il(e,t,a,os,n),l===s)break;s=l}s!==null&&a.stopPropagation()}else il(e,t,a,null,n)}}var os=null;function Bl(e,t,n,a){if(os=null,e=Io(a),e=_r(e),e!==null)if(t=Or(e),t===null)e=null;else if(n=t.tag,n===13){if(e=Sd(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null);return os=e,null}function Dd(e){switch(e){case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 1;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"toggle":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 4;case"message":switch(uf()){case Ro:return 1;case zd:return 4;case as:case pf:return 16;case Ed:return 536870912;default:return 16}default:return 16}}var sr=null,Do=null,Ga=null;function Od(){if(Ga)return Ga;var e,t=Do,n=t.length,a,s="value"in sr?sr.value:sr.textContent,l=s.length;for(e=0;e<n&&t[e]===s[e];e++);var o=n-e;for(a=1;a<=o&&t[n-a]===s[l-a];a++);return Ga=s.slice(e,1<a?1-a:void 0)}function Qa(e){var t=e.keyCode;return"charCode"in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function Ia(){return!0}function Di(){return!1}function pt(e){function t(n,a,s,l,o){this._reactName=n,this._targetInst=s,this.type=a,this.nativeEvent=l,this.target=o,this.currentTarget=null;for(var c in e)e.hasOwnProperty(c)&&(n=e[c],this[c]=n?n(l):l[c]);return this.isDefaultPrevented=(l.defaultPrevented!=null?l.defaultPrevented:l.returnValue===!1)?Ia:Di,this.isPropagationStopped=Di,this}return _e(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=Ia)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=Ia)},persist:function(){},isPersistent:Ia}),t}var gn={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},Oo=pt(gn),xa=_e({},gn,{view:0,detail:0}),Nf=pt(xa),Zs,el,Tn,Es=_e({},xa,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:Ao,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return"movementX"in e?e.movementX:(e!==Tn&&(Tn&&e.type==="mousemove"?(Zs=e.screenX-Tn.screenX,el=e.screenY-Tn.screenY):el=Zs=0,Tn=e),Zs)},movementY:function(e){return"movementY"in e?e.movementY:el}}),Oi=pt(Es),Cf=_e({},Es,{dataTransfer:0}),_f=pt(Cf),zf=_e({},xa,{relatedTarget:0}),tl=pt(zf),Ef=_e({},gn,{animationName:0,elapsedTime:0,pseudoElement:0}),Tf=pt(Ef),Pf=_e({},gn,{clipboardData:function(e){return"clipboardData"in e?e.clipboardData:window.clipboardData}}),If=pt(Pf),Rf=_e({},gn,{data:0}),Ai=pt(Rf),Mf={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},Ff={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},Lf={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function Df(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=Lf[e])?!!t[e]:!1}function Ao(){return Df}var Of=_e({},xa,{key:function(e){if(e.key){var t=Mf[e.key]||e.key;if(t!=="Unidentified")return t}return e.type==="keypress"?(e=Qa(e),e===13?"Enter":String.fromCharCode(e)):e.type==="keydown"||e.type==="keyup"?Ff[e.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:Ao,charCode:function(e){return e.type==="keypress"?Qa(e):0},keyCode:function(e){return e.type==="keydown"||e.type==="keyup"?e.keyCode:0},which:function(e){return e.type==="keypress"?Qa(e):e.type==="keydown"||e.type==="keyup"?e.keyCode:0}}),Af=pt(Of),$f=_e({},Es,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),$i=pt($f),Uf=_e({},xa,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:Ao}),Vf=pt(Uf),Bf=_e({},gn,{propertyName:0,elapsedTime:0,pseudoElement:0}),Wf=pt(Bf),Hf=_e({},Es,{deltaX:function(e){return"deltaX"in e?e.deltaX:"wheelDeltaX"in e?-e.wheelDeltaX:0},deltaY:function(e){return"deltaY"in e?e.deltaY:"wheelDeltaY"in e?-e.wheelDeltaY:"wheelDelta"in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0}),Gf=pt(Hf),Qf=[9,13,27,32],$o=Qt&&"CompositionEvent"in window,$n=null;Qt&&"documentMode"in document&&($n=document.documentMode);var qf=Qt&&"TextEvent"in window&&!$n,Ad=Qt&&(!$o||$n&&8<$n&&11>=$n),Ui=" ",Vi=!1;function $d(e,t){switch(e){case"keyup":return Qf.indexOf(t.keyCode)!==-1;case"keydown":return t.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function Ud(e){return e=e.detail,typeof e=="object"&&"data"in e?e.data:null}var Wr=!1;function Yf(e,t){switch(e){case"compositionend":return Ud(t);case"keypress":return t.which!==32?null:(Vi=!0,Ui);case"textInput":return e=t.data,e===Ui&&Vi?null:e;default:return null}}function Xf(e,t){if(Wr)return e==="compositionend"||!$o&&$d(e,t)?(e=Od(),Ga=Do=sr=null,Wr=!1,e):null;switch(e){case"paste":return null;case"keypress":if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case"compositionend":return Ad&&t.locale!=="ko"?null:t.data;default:return null}}var Kf={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function Bi(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t==="input"?!!Kf[e.type]:t==="textarea"}function Vd(e,t,n,a){yd(a),t=is(t,"onChange"),0<t.length&&(n=new Oo("onChange","change",null,n,a),e.push({event:n,listeners:t}))}var Un=null,Zn=null;function Jf(e){Zd(e,0)}function Ts(e){var t=Qr(e);if(pd(t))return e}function Zf(e,t){if(e==="change")return t}var Bd=!1;if(Qt){var rl;if(Qt){var nl="oninput"in document;if(!nl){var Wi=document.createElement("div");Wi.setAttribute("oninput","return;"),nl=typeof Wi.oninput=="function"}rl=nl}else rl=!1;Bd=rl&&(!document.documentMode||9<document.documentMode)}function Hi(){Un&&(Un.detachEvent("onpropertychange",Wd),Zn=Un=null)}function Wd(e){if(e.propertyName==="value"&&Ts(Zn)){var t=[];Vd(t,Zn,e,Io(e)),kd(Jf,t)}}function em(e,t,n){e==="focusin"?(Hi(),Un=t,Zn=n,Un.attachEvent("onpropertychange",Wd)):e==="focusout"&&Hi()}function tm(e){if(e==="selectionchange"||e==="keyup"||e==="keydown")return Ts(Zn)}function rm(e,t){if(e==="click")return Ts(t)}function nm(e,t){if(e==="input"||e==="change")return Ts(t)}function am(e,t){return e===t&&(e!==0||1/e===1/t)||e!==e&&t!==t}var Et=typeof Object.is=="function"?Object.is:am;function ea(e,t){if(Et(e,t))return!0;if(typeof e!="object"||e===null||typeof t!="object"||t===null)return!1;var n=Object.keys(e),a=Object.keys(t);if(n.length!==a.length)return!1;for(a=0;a<n.length;a++){var s=n[a];if(!Nl.call(t,s)||!Et(e[s],t[s]))return!1}return!0}function Gi(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function Qi(e,t){var n=Gi(e);e=0;for(var a;n;){if(n.nodeType===3){if(a=e+n.textContent.length,e<=t&&a>=t)return{node:n,offset:t-e};e=a}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=Gi(n)}}function Hd(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?Hd(e,t.parentNode):"contains"in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function Gd(){for(var e=window,t=ts();t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href=="string"}catch{n=!1}if(n)e=t.contentWindow;else break;t=ts(e.document)}return t}function Uo(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t==="input"&&(e.type==="text"||e.type==="search"||e.type==="tel"||e.type==="url"||e.type==="password")||t==="textarea"||e.contentEditable==="true")}function sm(e){var t=Gd(),n=e.focusedElem,a=e.selectionRange;if(t!==n&&n&&n.ownerDocument&&Hd(n.ownerDocument.documentElement,n)){if(a!==null&&Uo(n)){if(t=a.start,e=a.end,e===void 0&&(e=t),"selectionStart"in n)n.selectionStart=t,n.selectionEnd=Math.min(e,n.value.length);else if(e=(t=n.ownerDocument||document)&&t.defaultView||window,e.getSelection){e=e.getSelection();var s=n.textContent.length,l=Math.min(a.start,s);a=a.end===void 0?l:Math.min(a.end,s),!e.extend&&l>a&&(s=a,a=l,l=s),s=Qi(n,l);var o=Qi(n,a);s&&o&&(e.rangeCount!==1||e.anchorNode!==s.node||e.anchorOffset!==s.offset||e.focusNode!==o.node||e.focusOffset!==o.offset)&&(t=t.createRange(),t.setStart(s.node,s.offset),e.removeAllRanges(),l>a?(e.addRange(t),e.extend(o.node,o.offset)):(t.setEnd(o.node,o.offset),e.addRange(t)))}}for(t=[],e=n;e=e.parentNode;)e.nodeType===1&&t.push({element:e,left:e.scrollLeft,top:e.scrollTop});for(typeof n.focus=="function"&&n.focus(),n=0;n<t.length;n++)e=t[n],e.element.scrollLeft=e.left,e.element.scrollTop=e.top}}var lm=Qt&&"documentMode"in document&&11>=document.documentMode,Hr=null,Wl=null,Vn=null,Hl=!1;function qi(e,t,n){var a=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;Hl||Hr==null||Hr!==ts(a)||(a=Hr,"selectionStart"in a&&Uo(a)?a={start:a.selectionStart,end:a.selectionEnd}:(a=(a.ownerDocument&&a.ownerDocument.defaultView||window).getSelection(),a={anchorNode:a.anchorNode,anchorOffset:a.anchorOffset,focusNode:a.focusNode,focusOffset:a.focusOffset}),Vn&&ea(Vn,a)||(Vn=a,a=is(Wl,"onSelect"),0<a.length&&(t=new Oo("onSelect","select",null,t,n),e.push({event:t,listeners:a}),t.target=Hr)))}function Ra(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n["Webkit"+e]="webkit"+t,n["Moz"+e]="moz"+t,n}var Gr={animationend:Ra("Animation","AnimationEnd"),animationiteration:Ra("Animation","AnimationIteration"),animationstart:Ra("Animation","AnimationStart"),transitionend:Ra("Transition","TransitionEnd")},al={},Qd={};Qt&&(Qd=document.createElement("div").style,"AnimationEvent"in window||(delete Gr.animationend.animation,delete Gr.animationiteration.animation,delete Gr.animationstart.animation),"TransitionEvent"in window||delete Gr.transitionend.transition);function Ps(e){if(al[e])return al[e];if(!Gr[e])return e;var t=Gr[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in Qd)return al[e]=t[n];return e}var qd=Ps("animationend"),Yd=Ps("animationiteration"),Xd=Ps("animationstart"),Kd=Ps("transitionend"),Jd=new Map,Yi="abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");function jr(e,t){Jd.set(e,t),Dr(t,[e])}for(var sl=0;sl<Yi.length;sl++){var ll=Yi[sl],om=ll.toLowerCase(),im=ll[0].toUpperCase()+ll.slice(1);jr(om,"on"+im)}jr(qd,"onAnimationEnd");jr(Yd,"onAnimationIteration");jr(Xd,"onAnimationStart");jr("dblclick","onDoubleClick");jr("focusin","onFocus");jr("focusout","onBlur");jr(Kd,"onTransitionEnd");ln("onMouseEnter",["mouseout","mouseover"]);ln("onMouseLeave",["mouseout","mouseover"]);ln("onPointerEnter",["pointerout","pointerover"]);ln("onPointerLeave",["pointerout","pointerover"]);Dr("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));Dr("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));Dr("onBeforeInput",["compositionend","keypress","textInput","paste"]);Dr("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));Dr("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));Dr("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var Dn="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),cm=new Set("cancel close invalid load scroll toggle".split(" ").concat(Dn));function Xi(e,t,n){var a=e.type||"unknown-event";e.currentTarget=n,lf(a,t,void 0,e),e.currentTarget=null}function Zd(e,t){t=(t&4)!==0;for(var n=0;n<e.length;n++){var a=e[n],s=a.event;a=a.listeners;e:{var l=void 0;if(t)for(var o=a.length-1;0<=o;o--){var c=a[o],d=c.instance,p=c.currentTarget;if(c=c.listener,d!==l&&s.isPropagationStopped())break e;Xi(s,c,p),l=d}else for(o=0;o<a.length;o++){if(c=a[o],d=c.instance,p=c.currentTarget,c=c.listener,d!==l&&s.isPropagationStopped())break e;Xi(s,c,p),l=d}}}if(ns)throw e=$l,ns=!1,$l=null,e}function we(e,t){var n=t[Xl];n===void 0&&(n=t[Xl]=new Set);var a=e+"__bubble";n.has(a)||(eu(t,e,2,!1),n.add(a))}function ol(e,t,n){var a=0;t&&(a|=4),eu(n,e,a,t)}var Ma="_reactListening"+Math.random().toString(36).slice(2);function ta(e){if(!e[Ma]){e[Ma]=!0,od.forEach(function(n){n!=="selectionchange"&&(cm.has(n)||ol(n,!1,e),ol(n,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[Ma]||(t[Ma]=!0,ol("selectionchange",!1,t))}}function eu(e,t,n,a){switch(Dd(t)){case 1:var s=kf;break;case 4:s=Sf;break;default:s=Lo}n=s.bind(null,t,n,e),s=void 0,!Al||t!=="touchstart"&&t!=="touchmove"&&t!=="wheel"||(s=!0),a?s!==void 0?e.addEventListener(t,n,{capture:!0,passive:s}):e.addEventListener(t,n,!0):s!==void 0?e.addEventListener(t,n,{passive:s}):e.addEventListener(t,n,!1)}function il(e,t,n,a,s){var l=a;if(!(t&1)&&!(t&2)&&a!==null)e:for(;;){if(a===null)return;var o=a.tag;if(o===3||o===4){var c=a.stateNode.containerInfo;if(c===s||c.nodeType===8&&c.parentNode===s)break;if(o===4)for(o=a.return;o!==null;){var d=o.tag;if((d===3||d===4)&&(d=o.stateNode.containerInfo,d===s||d.nodeType===8&&d.parentNode===s))return;o=o.return}for(;c!==null;){if(o=_r(c),o===null)return;if(d=o.tag,d===5||d===6){a=l=o;continue e}c=c.parentNode}}a=a.return}kd(function(){var p=l,v=Io(n),g=[];e:{var x=Jd.get(e);if(x!==void 0){var k=Oo,w=e;switch(e){case"keypress":if(Qa(n)===0)break e;case"keydown":case"keyup":k=Af;break;case"focusin":w="focus",k=tl;break;case"focusout":w="blur",k=tl;break;case"beforeblur":case"afterblur":k=tl;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":k=Oi;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":k=_f;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":k=Vf;break;case qd:case Yd:case Xd:k=Tf;break;case Kd:k=Wf;break;case"scroll":k=Nf;break;case"wheel":k=Gf;break;case"copy":case"cut":case"paste":k=If;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":k=$i}var z=(t&4)!==0,F=!z&&e==="scroll",f=z?x!==null?x+"Capture":null:x;z=[];for(var u=p,h;u!==null;){h=u;var y=h.stateNode;if(h.tag===5&&y!==null&&(h=y,f!==null&&(y=Yn(u,f),y!=null&&z.push(ra(u,y,h)))),F)break;u=u.return}0<z.length&&(x=new k(x,w,null,n,v),g.push({event:x,listeners:z}))}}if(!(t&7)){e:{if(x=e==="mouseover"||e==="pointerover",k=e==="mouseout"||e==="pointerout",x&&n!==Dl&&(w=n.relatedTarget||n.fromElement)&&(_r(w)||w[qt]))break e;if((k||x)&&(x=v.window===v?v:(x=v.ownerDocument)?x.defaultView||x.parentWindow:window,k?(w=n.relatedTarget||n.toElement,k=p,w=w?_r(w):null,w!==null&&(F=Or(w),w!==F||w.tag!==5&&w.tag!==6)&&(w=null)):(k=null,w=p),k!==w)){if(z=Oi,y="onMouseLeave",f="onMouseEnter",u="mouse",(e==="pointerout"||e==="pointerover")&&(z=$i,y="onPointerLeave",f="onPointerEnter",u="pointer"),F=k==null?x:Qr(k),h=w==null?x:Qr(w),x=new z(y,u+"leave",k,n,v),x.target=F,x.relatedTarget=h,y=null,_r(v)===p&&(z=new z(f,u+"enter",w,n,v),z.target=h,z.relatedTarget=F,y=z),F=y,k&&w)t:{for(z=k,f=w,u=0,h=z;h;h=$r(h))u++;for(h=0,y=f;y;y=$r(y))h++;for(;0<u-h;)z=$r(z),u--;for(;0<h-u;)f=$r(f),h--;for(;u--;){if(z===f||f!==null&&z===f.alternate)break t;z=$r(z),f=$r(f)}z=null}else z=null;k!==null&&Ki(g,x,k,z,!1),w!==null&&F!==null&&Ki(g,F,w,z,!0)}}e:{if(x=p?Qr(p):window,k=x.nodeName&&x.nodeName.toLowerCase(),k==="select"||k==="input"&&x.type==="file")var j=Zf;else if(Bi(x))if(Bd)j=nm;else{j=tm;var I=em}else(k=x.nodeName)&&k.toLowerCase()==="input"&&(x.type==="checkbox"||x.type==="radio")&&(j=rm);if(j&&(j=j(e,p))){Vd(g,j,n,v);break e}I&&I(e,x,p),e==="focusout"&&(I=x._wrapperState)&&I.controlled&&x.type==="number"&&Il(x,"number",x.value)}switch(I=p?Qr(p):window,e){case"focusin":(Bi(I)||I.contentEditable==="true")&&(Hr=I,Wl=p,Vn=null);break;case"focusout":Vn=Wl=Hr=null;break;case"mousedown":Hl=!0;break;case"contextmenu":case"mouseup":case"dragend":Hl=!1,qi(g,n,v);break;case"selectionchange":if(lm)break;case"keydown":case"keyup":qi(g,n,v)}var _;if($o)e:{switch(e){case"compositionstart":var R="onCompositionStart";break e;case"compositionend":R="onCompositionEnd";break e;case"compositionupdate":R="onCompositionUpdate";break e}R=void 0}else Wr?$d(e,n)&&(R="onCompositionEnd"):e==="keydown"&&n.keyCode===229&&(R="onCompositionStart");R&&(Ad&&n.locale!=="ko"&&(Wr||R!=="onCompositionStart"?R==="onCompositionEnd"&&Wr&&(_=Od()):(sr=v,Do="value"in sr?sr.value:sr.textContent,Wr=!0)),I=is(p,R),0<I.length&&(R=new Ai(R,e,null,n,v),g.push({event:R,listeners:I}),_?R.data=_:(_=Ud(n),_!==null&&(R.data=_)))),(_=qf?Yf(e,n):Xf(e,n))&&(p=is(p,"onBeforeInput"),0<p.length&&(v=new Ai("onBeforeInput","beforeinput",null,n,v),g.push({event:v,listeners:p}),v.data=_))}Zd(g,t)})}function ra(e,t,n){return{instance:e,listener:t,currentTarget:n}}function is(e,t){for(var n=t+"Capture",a=[];e!==null;){var s=e,l=s.stateNode;s.tag===5&&l!==null&&(s=l,l=Yn(e,n),l!=null&&a.unshift(ra(e,l,s)),l=Yn(e,t),l!=null&&a.push(ra(e,l,s))),e=e.return}return a}function $r(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5);return e||null}function Ki(e,t,n,a,s){for(var l=t._reactName,o=[];n!==null&&n!==a;){var c=n,d=c.alternate,p=c.stateNode;if(d!==null&&d===a)break;c.tag===5&&p!==null&&(c=p,s?(d=Yn(n,l),d!=null&&o.unshift(ra(n,d,c))):s||(d=Yn(n,l),d!=null&&o.push(ra(n,d,c)))),n=n.return}o.length!==0&&e.push({event:t,listeners:o})}var dm=/\r\n?/g,um=/\u0000|\uFFFD/g;function Ji(e){return(typeof e=="string"?e:""+e).replace(dm,`
`).replace(um,"")}function Fa(e,t,n){if(t=Ji(t),Ji(e)!==t&&n)throw Error(B(425))}function cs(){}var Gl=null,Ql=null;function ql(e,t){return e==="textarea"||e==="noscript"||typeof t.children=="string"||typeof t.children=="number"||typeof t.dangerouslySetInnerHTML=="object"&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var Yl=typeof setTimeout=="function"?setTimeout:void 0,pm=typeof clearTimeout=="function"?clearTimeout:void 0,Zi=typeof Promise=="function"?Promise:void 0,fm=typeof queueMicrotask=="function"?queueMicrotask:typeof Zi<"u"?function(e){return Zi.resolve(null).then(e).catch(mm)}:Yl;function mm(e){setTimeout(function(){throw e})}function cl(e,t){var n=t,a=0;do{var s=n.nextSibling;if(e.removeChild(n),s&&s.nodeType===8)if(n=s.data,n==="/$"){if(a===0){e.removeChild(s),Jn(t);return}a--}else n!=="$"&&n!=="$?"&&n!=="$!"||a++;n=s}while(n);Jn(t)}function dr(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t==="$"||t==="$!"||t==="$?")break;if(t==="/$")return null}}return e}function ec(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="$"||n==="$!"||n==="$?"){if(t===0)return e;t--}else n==="/$"&&t++}e=e.previousSibling}return null}var vn=Math.random().toString(36).slice(2),Lt="__reactFiber$"+vn,na="__reactProps$"+vn,qt="__reactContainer$"+vn,Xl="__reactEvents$"+vn,hm="__reactListeners$"+vn,xm="__reactHandles$"+vn;function _r(e){var t=e[Lt];if(t)return t;for(var n=e.parentNode;n;){if(t=n[qt]||n[Lt]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=ec(e);e!==null;){if(n=e[Lt])return n;e=ec(e)}return t}e=n,n=e.parentNode}return null}function ga(e){return e=e[Lt]||e[qt],!e||e.tag!==5&&e.tag!==6&&e.tag!==13&&e.tag!==3?null:e}function Qr(e){if(e.tag===5||e.tag===6)return e.stateNode;throw Error(B(33))}function Is(e){return e[na]||null}var Kl=[],qr=-1;function br(e){return{current:e}}function ke(e){0>qr||(e.current=Kl[qr],Kl[qr]=null,qr--)}function be(e,t){qr++,Kl[qr]=e.current,e.current=t}var xr={},Be=br(xr),et=br(!1),Ir=xr;function on(e,t){var n=e.type.contextTypes;if(!n)return xr;var a=e.stateNode;if(a&&a.__reactInternalMemoizedUnmaskedChildContext===t)return a.__reactInternalMemoizedMaskedChildContext;var s={},l;for(l in n)s[l]=t[l];return a&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=t,e.__reactInternalMemoizedMaskedChildContext=s),s}function tt(e){return e=e.childContextTypes,e!=null}function ds(){ke(et),ke(Be)}function tc(e,t,n){if(Be.current!==xr)throw Error(B(168));be(Be,t),be(et,n)}function tu(e,t,n){var a=e.stateNode;if(t=t.childContextTypes,typeof a.getChildContext!="function")return n;a=a.getChildContext();for(var s in a)if(!(s in t))throw Error(B(108,Zp(e)||"Unknown",s));return _e({},n,a)}function us(e){return e=(e=e.stateNode)&&e.__reactInternalMemoizedMergedChildContext||xr,Ir=Be.current,be(Be,e),be(et,et.current),!0}function rc(e,t,n){var a=e.stateNode;if(!a)throw Error(B(169));n?(e=tu(e,t,Ir),a.__reactInternalMemoizedMergedChildContext=e,ke(et),ke(Be),be(Be,e)):ke(et),be(et,n)}var Ut=null,Rs=!1,dl=!1;function ru(e){Ut===null?Ut=[e]:Ut.push(e)}function gm(e){Rs=!0,ru(e)}function wr(){if(!dl&&Ut!==null){dl=!0;var e=0,t=je;try{var n=Ut;for(je=1;e<n.length;e++){var a=n[e];do a=a(!0);while(a!==null)}Ut=null,Rs=!1}catch(s){throw Ut!==null&&(Ut=Ut.slice(e+1)),_d(Ro,wr),s}finally{je=t,dl=!1}}return null}var Yr=[],Xr=0,ps=null,fs=0,mt=[],ht=0,Rr=null,Vt=1,Bt="";function Nr(e,t){Yr[Xr++]=fs,Yr[Xr++]=ps,ps=e,fs=t}function nu(e,t,n){mt[ht++]=Vt,mt[ht++]=Bt,mt[ht++]=Rr,Rr=e;var a=Vt;e=Bt;var s=32-_t(a)-1;a&=~(1<<s),n+=1;var l=32-_t(t)+s;if(30<l){var o=s-s%5;l=(a&(1<<o)-1).toString(32),a>>=o,s-=o,Vt=1<<32-_t(t)+s|n<<s|a,Bt=l+e}else Vt=1<<l|n<<s|a,Bt=e}function Vo(e){e.return!==null&&(Nr(e,1),nu(e,1,0))}function Bo(e){for(;e===ps;)ps=Yr[--Xr],Yr[Xr]=null,fs=Yr[--Xr],Yr[Xr]=null;for(;e===Rr;)Rr=mt[--ht],mt[ht]=null,Bt=mt[--ht],mt[ht]=null,Vt=mt[--ht],mt[ht]=null}var ct=null,it=null,Se=!1,Ct=null;function au(e,t){var n=xt(5,null,null,0);n.elementType="DELETED",n.stateNode=t,n.return=e,t=e.deletions,t===null?(e.deletions=[n],e.flags|=16):t.push(n)}function nc(e,t){switch(e.tag){case 5:var n=e.type;return t=t.nodeType!==1||n.toLowerCase()!==t.nodeName.toLowerCase()?null:t,t!==null?(e.stateNode=t,ct=e,it=dr(t.firstChild),!0):!1;case 6:return t=e.pendingProps===""||t.nodeType!==3?null:t,t!==null?(e.stateNode=t,ct=e,it=null,!0):!1;case 13:return t=t.nodeType!==8?null:t,t!==null?(n=Rr!==null?{id:Vt,overflow:Bt}:null,e.memoizedState={dehydrated:t,treeContext:n,retryLane:1073741824},n=xt(18,null,null,0),n.stateNode=t,n.return=e,e.child=n,ct=e,it=null,!0):!1;default:return!1}}function Jl(e){return(e.mode&1)!==0&&(e.flags&128)===0}function Zl(e){if(Se){var t=it;if(t){var n=t;if(!nc(e,t)){if(Jl(e))throw Error(B(418));t=dr(n.nextSibling);var a=ct;t&&nc(e,t)?au(a,n):(e.flags=e.flags&-4097|2,Se=!1,ct=e)}}else{if(Jl(e))throw Error(B(418));e.flags=e.flags&-4097|2,Se=!1,ct=e}}}function ac(e){for(e=e.return;e!==null&&e.tag!==5&&e.tag!==3&&e.tag!==13;)e=e.return;ct=e}function La(e){if(e!==ct)return!1;if(!Se)return ac(e),Se=!0,!1;var t;if((t=e.tag!==3)&&!(t=e.tag!==5)&&(t=e.type,t=t!=="head"&&t!=="body"&&!ql(e.type,e.memoizedProps)),t&&(t=it)){if(Jl(e))throw su(),Error(B(418));for(;t;)au(e,t),t=dr(t.nextSibling)}if(ac(e),e.tag===13){if(e=e.memoizedState,e=e!==null?e.dehydrated:null,!e)throw Error(B(317));e:{for(e=e.nextSibling,t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="/$"){if(t===0){it=dr(e.nextSibling);break e}t--}else n!=="$"&&n!=="$!"&&n!=="$?"||t++}e=e.nextSibling}it=null}}else it=ct?dr(e.stateNode.nextSibling):null;return!0}function su(){for(var e=it;e;)e=dr(e.nextSibling)}function cn(){it=ct=null,Se=!1}function Wo(e){Ct===null?Ct=[e]:Ct.push(e)}var vm=Jt.ReactCurrentBatchConfig;function Pn(e,t,n){if(e=n.ref,e!==null&&typeof e!="function"&&typeof e!="object"){if(n._owner){if(n=n._owner,n){if(n.tag!==1)throw Error(B(309));var a=n.stateNode}if(!a)throw Error(B(147,e));var s=a,l=""+e;return t!==null&&t.ref!==null&&typeof t.ref=="function"&&t.ref._stringRef===l?t.ref:(t=function(o){var c=s.refs;o===null?delete c[l]:c[l]=o},t._stringRef=l,t)}if(typeof e!="string")throw Error(B(284));if(!n._owner)throw Error(B(290,e))}return e}function Da(e,t){throw e=Object.prototype.toString.call(t),Error(B(31,e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e))}function sc(e){var t=e._init;return t(e._payload)}function lu(e){function t(f,u){if(e){var h=f.deletions;h===null?(f.deletions=[u],f.flags|=16):h.push(u)}}function n(f,u){if(!e)return null;for(;u!==null;)t(f,u),u=u.sibling;return null}function a(f,u){for(f=new Map;u!==null;)u.key!==null?f.set(u.key,u):f.set(u.index,u),u=u.sibling;return f}function s(f,u){return f=mr(f,u),f.index=0,f.sibling=null,f}function l(f,u,h){return f.index=h,e?(h=f.alternate,h!==null?(h=h.index,h<u?(f.flags|=2,u):h):(f.flags|=2,u)):(f.flags|=1048576,u)}function o(f){return e&&f.alternate===null&&(f.flags|=2),f}function c(f,u,h,y){return u===null||u.tag!==6?(u=gl(h,f.mode,y),u.return=f,u):(u=s(u,h),u.return=f,u)}function d(f,u,h,y){var j=h.type;return j===Br?v(f,u,h.props.children,y,h.key):u!==null&&(u.elementType===j||typeof j=="object"&&j!==null&&j.$$typeof===tr&&sc(j)===u.type)?(y=s(u,h.props),y.ref=Pn(f,u,h),y.return=f,y):(y=es(h.type,h.key,h.props,null,f.mode,y),y.ref=Pn(f,u,h),y.return=f,y)}function p(f,u,h,y){return u===null||u.tag!==4||u.stateNode.containerInfo!==h.containerInfo||u.stateNode.implementation!==h.implementation?(u=vl(h,f.mode,y),u.return=f,u):(u=s(u,h.children||[]),u.return=f,u)}function v(f,u,h,y,j){return u===null||u.tag!==7?(u=Pr(h,f.mode,y,j),u.return=f,u):(u=s(u,h),u.return=f,u)}function g(f,u,h){if(typeof u=="string"&&u!==""||typeof u=="number")return u=gl(""+u,f.mode,h),u.return=f,u;if(typeof u=="object"&&u!==null){switch(u.$$typeof){case Ca:return h=es(u.type,u.key,u.props,null,f.mode,h),h.ref=Pn(f,null,u),h.return=f,h;case Vr:return u=vl(u,f.mode,h),u.return=f,u;case tr:var y=u._init;return g(f,y(u._payload),h)}if(Fn(u)||Cn(u))return u=Pr(u,f.mode,h,null),u.return=f,u;Da(f,u)}return null}function x(f,u,h,y){var j=u!==null?u.key:null;if(typeof h=="string"&&h!==""||typeof h=="number")return j!==null?null:c(f,u,""+h,y);if(typeof h=="object"&&h!==null){switch(h.$$typeof){case Ca:return h.key===j?d(f,u,h,y):null;case Vr:return h.key===j?p(f,u,h,y):null;case tr:return j=h._init,x(f,u,j(h._payload),y)}if(Fn(h)||Cn(h))return j!==null?null:v(f,u,h,y,null);Da(f,h)}return null}function k(f,u,h,y,j){if(typeof y=="string"&&y!==""||typeof y=="number")return f=f.get(h)||null,c(u,f,""+y,j);if(typeof y=="object"&&y!==null){switch(y.$$typeof){case Ca:return f=f.get(y.key===null?h:y.key)||null,d(u,f,y,j);case Vr:return f=f.get(y.key===null?h:y.key)||null,p(u,f,y,j);case tr:var I=y._init;return k(f,u,h,I(y._payload),j)}if(Fn(y)||Cn(y))return f=f.get(h)||null,v(u,f,y,j,null);Da(u,y)}return null}function w(f,u,h,y){for(var j=null,I=null,_=u,R=u=0,G=null;_!==null&&R<h.length;R++){_.index>R?(G=_,_=null):G=_.sibling;var W=x(f,_,h[R],y);if(W===null){_===null&&(_=G);break}e&&_&&W.alternate===null&&t(f,_),u=l(W,u,R),I===null?j=W:I.sibling=W,I=W,_=G}if(R===h.length)return n(f,_),Se&&Nr(f,R),j;if(_===null){for(;R<h.length;R++)_=g(f,h[R],y),_!==null&&(u=l(_,u,R),I===null?j=_:I.sibling=_,I=_);return Se&&Nr(f,R),j}for(_=a(f,_);R<h.length;R++)G=k(_,f,R,h[R],y),G!==null&&(e&&G.alternate!==null&&_.delete(G.key===null?R:G.key),u=l(G,u,R),I===null?j=G:I.sibling=G,I=G);return e&&_.forEach(function(b){return t(f,b)}),Se&&Nr(f,R),j}function z(f,u,h,y){var j=Cn(h);if(typeof j!="function")throw Error(B(150));if(h=j.call(h),h==null)throw Error(B(151));for(var I=j=null,_=u,R=u=0,G=null,W=h.next();_!==null&&!W.done;R++,W=h.next()){_.index>R?(G=_,_=null):G=_.sibling;var b=x(f,_,W.value,y);if(b===null){_===null&&(_=G);break}e&&_&&b.alternate===null&&t(f,_),u=l(b,u,R),I===null?j=b:I.sibling=b,I=b,_=G}if(W.done)return n(f,_),Se&&Nr(f,R),j;if(_===null){for(;!W.done;R++,W=h.next())W=g(f,W.value,y),W!==null&&(u=l(W,u,R),I===null?j=W:I.sibling=W,I=W);return Se&&Nr(f,R),j}for(_=a(f,_);!W.done;R++,W=h.next())W=k(_,f,R,W.value,y),W!==null&&(e&&W.alternate!==null&&_.delete(W.key===null?R:W.key),u=l(W,u,R),I===null?j=W:I.sibling=W,I=W);return e&&_.forEach(function(N){return t(f,N)}),Se&&Nr(f,R),j}function F(f,u,h,y){if(typeof h=="object"&&h!==null&&h.type===Br&&h.key===null&&(h=h.props.children),typeof h=="object"&&h!==null){switch(h.$$typeof){case Ca:e:{for(var j=h.key,I=u;I!==null;){if(I.key===j){if(j=h.type,j===Br){if(I.tag===7){n(f,I.sibling),u=s(I,h.props.children),u.return=f,f=u;break e}}else if(I.elementType===j||typeof j=="object"&&j!==null&&j.$$typeof===tr&&sc(j)===I.type){n(f,I.sibling),u=s(I,h.props),u.ref=Pn(f,I,h),u.return=f,f=u;break e}n(f,I);break}else t(f,I);I=I.sibling}h.type===Br?(u=Pr(h.props.children,f.mode,y,h.key),u.return=f,f=u):(y=es(h.type,h.key,h.props,null,f.mode,y),y.ref=Pn(f,u,h),y.return=f,f=y)}return o(f);case Vr:e:{for(I=h.key;u!==null;){if(u.key===I)if(u.tag===4&&u.stateNode.containerInfo===h.containerInfo&&u.stateNode.implementation===h.implementation){n(f,u.sibling),u=s(u,h.children||[]),u.return=f,f=u;break e}else{n(f,u);break}else t(f,u);u=u.sibling}u=vl(h,f.mode,y),u.return=f,f=u}return o(f);case tr:return I=h._init,F(f,u,I(h._payload),y)}if(Fn(h))return w(f,u,h,y);if(Cn(h))return z(f,u,h,y);Da(f,h)}return typeof h=="string"&&h!==""||typeof h=="number"?(h=""+h,u!==null&&u.tag===6?(n(f,u.sibling),u=s(u,h),u.return=f,f=u):(n(f,u),u=gl(h,f.mode,y),u.return=f,f=u),o(f)):n(f,u)}return F}var dn=lu(!0),ou=lu(!1),ms=br(null),hs=null,Kr=null,Ho=null;function Go(){Ho=Kr=hs=null}function Qo(e){var t=ms.current;ke(ms),e._currentValue=t}function eo(e,t,n){for(;e!==null;){var a=e.alternate;if((e.childLanes&t)!==t?(e.childLanes|=t,a!==null&&(a.childLanes|=t)):a!==null&&(a.childLanes&t)!==t&&(a.childLanes|=t),e===n)break;e=e.return}}function an(e,t){hs=e,Ho=Kr=null,e=e.dependencies,e!==null&&e.firstContext!==null&&(e.lanes&t&&(Ze=!0),e.firstContext=null)}function yt(e){var t=e._currentValue;if(Ho!==e)if(e={context:e,memoizedValue:t,next:null},Kr===null){if(hs===null)throw Error(B(308));Kr=e,hs.dependencies={lanes:0,firstContext:e}}else Kr=Kr.next=e;return t}var zr=null;function qo(e){zr===null?zr=[e]:zr.push(e)}function iu(e,t,n,a){var s=t.interleaved;return s===null?(n.next=n,qo(t)):(n.next=s.next,s.next=n),t.interleaved=n,Yt(e,a)}function Yt(e,t){e.lanes|=t;var n=e.alternate;for(n!==null&&(n.lanes|=t),n=e,e=e.return;e!==null;)e.childLanes|=t,n=e.alternate,n!==null&&(n.childLanes|=t),n=e,e=e.return;return n.tag===3?n.stateNode:null}var rr=!1;function Yo(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,interleaved:null,lanes:0},effects:null}}function cu(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,effects:e.effects})}function Ht(e,t){return{eventTime:e,lane:t,tag:0,payload:null,callback:null,next:null}}function ur(e,t,n){var a=e.updateQueue;if(a===null)return null;if(a=a.shared,ge&2){var s=a.pending;return s===null?t.next=t:(t.next=s.next,s.next=t),a.pending=t,Yt(e,n)}return s=a.interleaved,s===null?(t.next=t,qo(a)):(t.next=s.next,s.next=t),a.interleaved=t,Yt(e,n)}function qa(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,(n&4194240)!==0)){var a=t.lanes;a&=e.pendingLanes,n|=a,t.lanes=n,Mo(e,n)}}function lc(e,t){var n=e.updateQueue,a=e.alternate;if(a!==null&&(a=a.updateQueue,n===a)){var s=null,l=null;if(n=n.firstBaseUpdate,n!==null){do{var o={eventTime:n.eventTime,lane:n.lane,tag:n.tag,payload:n.payload,callback:n.callback,next:null};l===null?s=l=o:l=l.next=o,n=n.next}while(n!==null);l===null?s=l=t:l=l.next=t}else s=l=t;n={baseState:a.baseState,firstBaseUpdate:s,lastBaseUpdate:l,shared:a.shared,effects:a.effects},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}function xs(e,t,n,a){var s=e.updateQueue;rr=!1;var l=s.firstBaseUpdate,o=s.lastBaseUpdate,c=s.shared.pending;if(c!==null){s.shared.pending=null;var d=c,p=d.next;d.next=null,o===null?l=p:o.next=p,o=d;var v=e.alternate;v!==null&&(v=v.updateQueue,c=v.lastBaseUpdate,c!==o&&(c===null?v.firstBaseUpdate=p:c.next=p,v.lastBaseUpdate=d))}if(l!==null){var g=s.baseState;o=0,v=p=d=null,c=l;do{var x=c.lane,k=c.eventTime;if((a&x)===x){v!==null&&(v=v.next={eventTime:k,lane:0,tag:c.tag,payload:c.payload,callback:c.callback,next:null});e:{var w=e,z=c;switch(x=t,k=n,z.tag){case 1:if(w=z.payload,typeof w=="function"){g=w.call(k,g,x);break e}g=w;break e;case 3:w.flags=w.flags&-65537|128;case 0:if(w=z.payload,x=typeof w=="function"?w.call(k,g,x):w,x==null)break e;g=_e({},g,x);break e;case 2:rr=!0}}c.callback!==null&&c.lane!==0&&(e.flags|=64,x=s.effects,x===null?s.effects=[c]:x.push(c))}else k={eventTime:k,lane:x,tag:c.tag,payload:c.payload,callback:c.callback,next:null},v===null?(p=v=k,d=g):v=v.next=k,o|=x;if(c=c.next,c===null){if(c=s.shared.pending,c===null)break;x=c,c=x.next,x.next=null,s.lastBaseUpdate=x,s.shared.pending=null}}while(!0);if(v===null&&(d=g),s.baseState=d,s.firstBaseUpdate=p,s.lastBaseUpdate=v,t=s.shared.interleaved,t!==null){s=t;do o|=s.lane,s=s.next;while(s!==t)}else l===null&&(s.shared.lanes=0);Fr|=o,e.lanes=o,e.memoizedState=g}}function oc(e,t,n){if(e=t.effects,t.effects=null,e!==null)for(t=0;t<e.length;t++){var a=e[t],s=a.callback;if(s!==null){if(a.callback=null,a=n,typeof s!="function")throw Error(B(191,s));s.call(a)}}}var va={},Ot=br(va),aa=br(va),sa=br(va);function Er(e){if(e===va)throw Error(B(174));return e}function Xo(e,t){switch(be(sa,t),be(aa,e),be(Ot,va),e=t.nodeType,e){case 9:case 11:t=(t=t.documentElement)?t.namespaceURI:Ml(null,"");break;default:e=e===8?t.parentNode:t,t=e.namespaceURI||null,e=e.tagName,t=Ml(t,e)}ke(Ot),be(Ot,t)}function un(){ke(Ot),ke(aa),ke(sa)}function du(e){Er(sa.current);var t=Er(Ot.current),n=Ml(t,e.type);t!==n&&(be(aa,e),be(Ot,n))}function Ko(e){aa.current===e&&(ke(Ot),ke(aa))}var Ne=br(0);function gs(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||n.data==="$?"||n.data==="$!"))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==void 0){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var ul=[];function Jo(){for(var e=0;e<ul.length;e++)ul[e]._workInProgressVersionPrimary=null;ul.length=0}var Ya=Jt.ReactCurrentDispatcher,pl=Jt.ReactCurrentBatchConfig,Mr=0,Ce=null,Ie=null,Me=null,vs=!1,Bn=!1,la=0,ym=0;function $e(){throw Error(B(321))}function Zo(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!Et(e[n],t[n]))return!1;return!0}function ei(e,t,n,a,s,l){if(Mr=l,Ce=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,Ya.current=e===null||e.memoizedState===null?km:Sm,e=n(a,s),Bn){l=0;do{if(Bn=!1,la=0,25<=l)throw Error(B(301));l+=1,Me=Ie=null,t.updateQueue=null,Ya.current=Nm,e=n(a,s)}while(Bn)}if(Ya.current=ys,t=Ie!==null&&Ie.next!==null,Mr=0,Me=Ie=Ce=null,vs=!1,t)throw Error(B(300));return e}function ti(){var e=la!==0;return la=0,e}function Ft(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return Me===null?Ce.memoizedState=Me=e:Me=Me.next=e,Me}function jt(){if(Ie===null){var e=Ce.alternate;e=e!==null?e.memoizedState:null}else e=Ie.next;var t=Me===null?Ce.memoizedState:Me.next;if(t!==null)Me=t,Ie=e;else{if(e===null)throw Error(B(310));Ie=e,e={memoizedState:Ie.memoizedState,baseState:Ie.baseState,baseQueue:Ie.baseQueue,queue:Ie.queue,next:null},Me===null?Ce.memoizedState=Me=e:Me=Me.next=e}return Me}function oa(e,t){return typeof t=="function"?t(e):t}function fl(e){var t=jt(),n=t.queue;if(n===null)throw Error(B(311));n.lastRenderedReducer=e;var a=Ie,s=a.baseQueue,l=n.pending;if(l!==null){if(s!==null){var o=s.next;s.next=l.next,l.next=o}a.baseQueue=s=l,n.pending=null}if(s!==null){l=s.next,a=a.baseState;var c=o=null,d=null,p=l;do{var v=p.lane;if((Mr&v)===v)d!==null&&(d=d.next={lane:0,action:p.action,hasEagerState:p.hasEagerState,eagerState:p.eagerState,next:null}),a=p.hasEagerState?p.eagerState:e(a,p.action);else{var g={lane:v,action:p.action,hasEagerState:p.hasEagerState,eagerState:p.eagerState,next:null};d===null?(c=d=g,o=a):d=d.next=g,Ce.lanes|=v,Fr|=v}p=p.next}while(p!==null&&p!==l);d===null?o=a:d.next=c,Et(a,t.memoizedState)||(Ze=!0),t.memoizedState=a,t.baseState=o,t.baseQueue=d,n.lastRenderedState=a}if(e=n.interleaved,e!==null){s=e;do l=s.lane,Ce.lanes|=l,Fr|=l,s=s.next;while(s!==e)}else s===null&&(n.lanes=0);return[t.memoizedState,n.dispatch]}function ml(e){var t=jt(),n=t.queue;if(n===null)throw Error(B(311));n.lastRenderedReducer=e;var a=n.dispatch,s=n.pending,l=t.memoizedState;if(s!==null){n.pending=null;var o=s=s.next;do l=e(l,o.action),o=o.next;while(o!==s);Et(l,t.memoizedState)||(Ze=!0),t.memoizedState=l,t.baseQueue===null&&(t.baseState=l),n.lastRenderedState=l}return[l,a]}function uu(){}function pu(e,t){var n=Ce,a=jt(),s=t(),l=!Et(a.memoizedState,s);if(l&&(a.memoizedState=s,Ze=!0),a=a.queue,ri(hu.bind(null,n,a,e),[e]),a.getSnapshot!==t||l||Me!==null&&Me.memoizedState.tag&1){if(n.flags|=2048,ia(9,mu.bind(null,n,a,s,t),void 0,null),Fe===null)throw Error(B(349));Mr&30||fu(n,t,s)}return s}function fu(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=Ce.updateQueue,t===null?(t={lastEffect:null,stores:null},Ce.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function mu(e,t,n,a){t.value=n,t.getSnapshot=a,xu(t)&&gu(e)}function hu(e,t,n){return n(function(){xu(t)&&gu(e)})}function xu(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!Et(e,n)}catch{return!0}}function gu(e){var t=Yt(e,1);t!==null&&zt(t,e,1,-1)}function ic(e){var t=Ft();return typeof e=="function"&&(e=e()),t.memoizedState=t.baseState=e,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:oa,lastRenderedState:e},t.queue=e,e=e.dispatch=wm.bind(null,Ce,e),[t.memoizedState,e]}function ia(e,t,n,a){return e={tag:e,create:t,destroy:n,deps:a,next:null},t=Ce.updateQueue,t===null?(t={lastEffect:null,stores:null},Ce.updateQueue=t,t.lastEffect=e.next=e):(n=t.lastEffect,n===null?t.lastEffect=e.next=e:(a=n.next,n.next=e,e.next=a,t.lastEffect=e)),e}function vu(){return jt().memoizedState}function Xa(e,t,n,a){var s=Ft();Ce.flags|=e,s.memoizedState=ia(1|t,n,void 0,a===void 0?null:a)}function Ms(e,t,n,a){var s=jt();a=a===void 0?null:a;var l=void 0;if(Ie!==null){var o=Ie.memoizedState;if(l=o.destroy,a!==null&&Zo(a,o.deps)){s.memoizedState=ia(t,n,l,a);return}}Ce.flags|=e,s.memoizedState=ia(1|t,n,l,a)}function cc(e,t){return Xa(8390656,8,e,t)}function ri(e,t){return Ms(2048,8,e,t)}function yu(e,t){return Ms(4,2,e,t)}function ju(e,t){return Ms(4,4,e,t)}function bu(e,t){if(typeof t=="function")return e=e(),t(e),function(){t(null)};if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function wu(e,t,n){return n=n!=null?n.concat([e]):null,Ms(4,4,bu.bind(null,t,e),n)}function ni(){}function ku(e,t){var n=jt();t=t===void 0?null:t;var a=n.memoizedState;return a!==null&&t!==null&&Zo(t,a[1])?a[0]:(n.memoizedState=[e,t],e)}function Su(e,t){var n=jt();t=t===void 0?null:t;var a=n.memoizedState;return a!==null&&t!==null&&Zo(t,a[1])?a[0]:(e=e(),n.memoizedState=[e,t],e)}function Nu(e,t,n){return Mr&21?(Et(n,t)||(n=Td(),Ce.lanes|=n,Fr|=n,e.baseState=!0),t):(e.baseState&&(e.baseState=!1,Ze=!0),e.memoizedState=n)}function jm(e,t){var n=je;je=n!==0&&4>n?n:4,e(!0);var a=pl.transition;pl.transition={};try{e(!1),t()}finally{je=n,pl.transition=a}}function Cu(){return jt().memoizedState}function bm(e,t,n){var a=fr(e);if(n={lane:a,action:n,hasEagerState:!1,eagerState:null,next:null},_u(e))zu(t,n);else if(n=iu(e,t,n,a),n!==null){var s=qe();zt(n,e,a,s),Eu(n,t,a)}}function wm(e,t,n){var a=fr(e),s={lane:a,action:n,hasEagerState:!1,eagerState:null,next:null};if(_u(e))zu(t,s);else{var l=e.alternate;if(e.lanes===0&&(l===null||l.lanes===0)&&(l=t.lastRenderedReducer,l!==null))try{var o=t.lastRenderedState,c=l(o,n);if(s.hasEagerState=!0,s.eagerState=c,Et(c,o)){var d=t.interleaved;d===null?(s.next=s,qo(t)):(s.next=d.next,d.next=s),t.interleaved=s;return}}catch{}finally{}n=iu(e,t,s,a),n!==null&&(s=qe(),zt(n,e,a,s),Eu(n,t,a))}}function _u(e){var t=e.alternate;return e===Ce||t!==null&&t===Ce}function zu(e,t){Bn=vs=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function Eu(e,t,n){if(n&4194240){var a=t.lanes;a&=e.pendingLanes,n|=a,t.lanes=n,Mo(e,n)}}var ys={readContext:yt,useCallback:$e,useContext:$e,useEffect:$e,useImperativeHandle:$e,useInsertionEffect:$e,useLayoutEffect:$e,useMemo:$e,useReducer:$e,useRef:$e,useState:$e,useDebugValue:$e,useDeferredValue:$e,useTransition:$e,useMutableSource:$e,useSyncExternalStore:$e,useId:$e,unstable_isNewReconciler:!1},km={readContext:yt,useCallback:function(e,t){return Ft().memoizedState=[e,t===void 0?null:t],e},useContext:yt,useEffect:cc,useImperativeHandle:function(e,t,n){return n=n!=null?n.concat([e]):null,Xa(4194308,4,bu.bind(null,t,e),n)},useLayoutEffect:function(e,t){return Xa(4194308,4,e,t)},useInsertionEffect:function(e,t){return Xa(4,2,e,t)},useMemo:function(e,t){var n=Ft();return t=t===void 0?null:t,e=e(),n.memoizedState=[e,t],e},useReducer:function(e,t,n){var a=Ft();return t=n!==void 0?n(t):t,a.memoizedState=a.baseState=t,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:t},a.queue=e,e=e.dispatch=bm.bind(null,Ce,e),[a.memoizedState,e]},useRef:function(e){var t=Ft();return e={current:e},t.memoizedState=e},useState:ic,useDebugValue:ni,useDeferredValue:function(e){return Ft().memoizedState=e},useTransition:function(){var e=ic(!1),t=e[0];return e=jm.bind(null,e[1]),Ft().memoizedState=e,[t,e]},useMutableSource:function(){},useSyncExternalStore:function(e,t,n){var a=Ce,s=Ft();if(Se){if(n===void 0)throw Error(B(407));n=n()}else{if(n=t(),Fe===null)throw Error(B(349));Mr&30||fu(a,t,n)}s.memoizedState=n;var l={value:n,getSnapshot:t};return s.queue=l,cc(hu.bind(null,a,l,e),[e]),a.flags|=2048,ia(9,mu.bind(null,a,l,n,t),void 0,null),n},useId:function(){var e=Ft(),t=Fe.identifierPrefix;if(Se){var n=Bt,a=Vt;n=(a&~(1<<32-_t(a)-1)).toString(32)+n,t=":"+t+"R"+n,n=la++,0<n&&(t+="H"+n.toString(32)),t+=":"}else n=ym++,t=":"+t+"r"+n.toString(32)+":";return e.memoizedState=t},unstable_isNewReconciler:!1},Sm={readContext:yt,useCallback:ku,useContext:yt,useEffect:ri,useImperativeHandle:wu,useInsertionEffect:yu,useLayoutEffect:ju,useMemo:Su,useReducer:fl,useRef:vu,useState:function(){return fl(oa)},useDebugValue:ni,useDeferredValue:function(e){var t=jt();return Nu(t,Ie.memoizedState,e)},useTransition:function(){var e=fl(oa)[0],t=jt().memoizedState;return[e,t]},useMutableSource:uu,useSyncExternalStore:pu,useId:Cu,unstable_isNewReconciler:!1},Nm={readContext:yt,useCallback:ku,useContext:yt,useEffect:ri,useImperativeHandle:wu,useInsertionEffect:yu,useLayoutEffect:ju,useMemo:Su,useReducer:ml,useRef:vu,useState:function(){return ml(oa)},useDebugValue:ni,useDeferredValue:function(e){var t=jt();return Ie===null?t.memoizedState=e:Nu(t,Ie.memoizedState,e)},useTransition:function(){var e=ml(oa)[0],t=jt().memoizedState;return[e,t]},useMutableSource:uu,useSyncExternalStore:pu,useId:Cu,unstable_isNewReconciler:!1};function St(e,t){if(e&&e.defaultProps){t=_e({},t),e=e.defaultProps;for(var n in e)t[n]===void 0&&(t[n]=e[n]);return t}return t}function to(e,t,n,a){t=e.memoizedState,n=n(a,t),n=n==null?t:_e({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var Fs={isMounted:function(e){return(e=e._reactInternals)?Or(e)===e:!1},enqueueSetState:function(e,t,n){e=e._reactInternals;var a=qe(),s=fr(e),l=Ht(a,s);l.payload=t,n!=null&&(l.callback=n),t=ur(e,l,s),t!==null&&(zt(t,e,s,a),qa(t,e,s))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var a=qe(),s=fr(e),l=Ht(a,s);l.tag=1,l.payload=t,n!=null&&(l.callback=n),t=ur(e,l,s),t!==null&&(zt(t,e,s,a),qa(t,e,s))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=qe(),a=fr(e),s=Ht(n,a);s.tag=2,t!=null&&(s.callback=t),t=ur(e,s,a),t!==null&&(zt(t,e,a,n),qa(t,e,a))}};function dc(e,t,n,a,s,l,o){return e=e.stateNode,typeof e.shouldComponentUpdate=="function"?e.shouldComponentUpdate(a,l,o):t.prototype&&t.prototype.isPureReactComponent?!ea(n,a)||!ea(s,l):!0}function Tu(e,t,n){var a=!1,s=xr,l=t.contextType;return typeof l=="object"&&l!==null?l=yt(l):(s=tt(t)?Ir:Be.current,a=t.contextTypes,l=(a=a!=null)?on(e,s):xr),t=new t(n,l),e.memoizedState=t.state!==null&&t.state!==void 0?t.state:null,t.updater=Fs,e.stateNode=t,t._reactInternals=e,a&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=s,e.__reactInternalMemoizedMaskedChildContext=l),t}function uc(e,t,n,a){e=t.state,typeof t.componentWillReceiveProps=="function"&&t.componentWillReceiveProps(n,a),typeof t.UNSAFE_componentWillReceiveProps=="function"&&t.UNSAFE_componentWillReceiveProps(n,a),t.state!==e&&Fs.enqueueReplaceState(t,t.state,null)}function ro(e,t,n,a){var s=e.stateNode;s.props=n,s.state=e.memoizedState,s.refs={},Yo(e);var l=t.contextType;typeof l=="object"&&l!==null?s.context=yt(l):(l=tt(t)?Ir:Be.current,s.context=on(e,l)),s.state=e.memoizedState,l=t.getDerivedStateFromProps,typeof l=="function"&&(to(e,t,l,n),s.state=e.memoizedState),typeof t.getDerivedStateFromProps=="function"||typeof s.getSnapshotBeforeUpdate=="function"||typeof s.UNSAFE_componentWillMount!="function"&&typeof s.componentWillMount!="function"||(t=s.state,typeof s.componentWillMount=="function"&&s.componentWillMount(),typeof s.UNSAFE_componentWillMount=="function"&&s.UNSAFE_componentWillMount(),t!==s.state&&Fs.enqueueReplaceState(s,s.state,null),xs(e,n,s,a),s.state=e.memoizedState),typeof s.componentDidMount=="function"&&(e.flags|=4194308)}function pn(e,t){try{var n="",a=t;do n+=Jp(a),a=a.return;while(a);var s=n}catch(l){s=`
Error generating stack: `+l.message+`
`+l.stack}return{value:e,source:t,stack:s,digest:null}}function hl(e,t,n){return{value:e,source:null,stack:n??null,digest:t??null}}function no(e,t){try{console.error(t.value)}catch(n){setTimeout(function(){throw n})}}var Cm=typeof WeakMap=="function"?WeakMap:Map;function Pu(e,t,n){n=Ht(-1,n),n.tag=3,n.payload={element:null};var a=t.value;return n.callback=function(){bs||(bs=!0,mo=a),no(e,t)},n}function Iu(e,t,n){n=Ht(-1,n),n.tag=3;var a=e.type.getDerivedStateFromError;if(typeof a=="function"){var s=t.value;n.payload=function(){return a(s)},n.callback=function(){no(e,t)}}var l=e.stateNode;return l!==null&&typeof l.componentDidCatch=="function"&&(n.callback=function(){no(e,t),typeof a!="function"&&(pr===null?pr=new Set([this]):pr.add(this));var o=t.stack;this.componentDidCatch(t.value,{componentStack:o!==null?o:""})}),n}function pc(e,t,n){var a=e.pingCache;if(a===null){a=e.pingCache=new Cm;var s=new Set;a.set(t,s)}else s=a.get(t),s===void 0&&(s=new Set,a.set(t,s));s.has(n)||(s.add(n),e=$m.bind(null,e,t,n),t.then(e,e))}function fc(e){do{var t;if((t=e.tag===13)&&(t=e.memoizedState,t=t!==null?t.dehydrated!==null:!0),t)return e;e=e.return}while(e!==null);return null}function mc(e,t,n,a,s){return e.mode&1?(e.flags|=65536,e.lanes=s,e):(e===t?e.flags|=65536:(e.flags|=128,n.flags|=131072,n.flags&=-52805,n.tag===1&&(n.alternate===null?n.tag=17:(t=Ht(-1,1),t.tag=2,ur(n,t,1))),n.lanes|=1),e)}var _m=Jt.ReactCurrentOwner,Ze=!1;function Ge(e,t,n,a){t.child=e===null?ou(t,null,n,a):dn(t,e.child,n,a)}function hc(e,t,n,a,s){n=n.render;var l=t.ref;return an(t,s),a=ei(e,t,n,a,l,s),n=ti(),e!==null&&!Ze?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~s,Xt(e,t,s)):(Se&&n&&Vo(t),t.flags|=1,Ge(e,t,a,s),t.child)}function xc(e,t,n,a,s){if(e===null){var l=n.type;return typeof l=="function"&&!ui(l)&&l.defaultProps===void 0&&n.compare===null&&n.defaultProps===void 0?(t.tag=15,t.type=l,Ru(e,t,l,a,s)):(e=es(n.type,null,a,t,t.mode,s),e.ref=t.ref,e.return=t,t.child=e)}if(l=e.child,!(e.lanes&s)){var o=l.memoizedProps;if(n=n.compare,n=n!==null?n:ea,n(o,a)&&e.ref===t.ref)return Xt(e,t,s)}return t.flags|=1,e=mr(l,a),e.ref=t.ref,e.return=t,t.child=e}function Ru(e,t,n,a,s){if(e!==null){var l=e.memoizedProps;if(ea(l,a)&&e.ref===t.ref)if(Ze=!1,t.pendingProps=a=l,(e.lanes&s)!==0)e.flags&131072&&(Ze=!0);else return t.lanes=e.lanes,Xt(e,t,s)}return ao(e,t,n,a,s)}function Mu(e,t,n){var a=t.pendingProps,s=a.children,l=e!==null?e.memoizedState:null;if(a.mode==="hidden")if(!(t.mode&1))t.memoizedState={baseLanes:0,cachePool:null,transitions:null},be(Zr,ot),ot|=n;else{if(!(n&1073741824))return e=l!==null?l.baseLanes|n:n,t.lanes=t.childLanes=1073741824,t.memoizedState={baseLanes:e,cachePool:null,transitions:null},t.updateQueue=null,be(Zr,ot),ot|=e,null;t.memoizedState={baseLanes:0,cachePool:null,transitions:null},a=l!==null?l.baseLanes:n,be(Zr,ot),ot|=a}else l!==null?(a=l.baseLanes|n,t.memoizedState=null):a=n,be(Zr,ot),ot|=a;return Ge(e,t,s,n),t.child}function Fu(e,t){var n=t.ref;(e===null&&n!==null||e!==null&&e.ref!==n)&&(t.flags|=512,t.flags|=2097152)}function ao(e,t,n,a,s){var l=tt(n)?Ir:Be.current;return l=on(t,l),an(t,s),n=ei(e,t,n,a,l,s),a=ti(),e!==null&&!Ze?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~s,Xt(e,t,s)):(Se&&a&&Vo(t),t.flags|=1,Ge(e,t,n,s),t.child)}function gc(e,t,n,a,s){if(tt(n)){var l=!0;us(t)}else l=!1;if(an(t,s),t.stateNode===null)Ka(e,t),Tu(t,n,a),ro(t,n,a,s),a=!0;else if(e===null){var o=t.stateNode,c=t.memoizedProps;o.props=c;var d=o.context,p=n.contextType;typeof p=="object"&&p!==null?p=yt(p):(p=tt(n)?Ir:Be.current,p=on(t,p));var v=n.getDerivedStateFromProps,g=typeof v=="function"||typeof o.getSnapshotBeforeUpdate=="function";g||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(c!==a||d!==p)&&uc(t,o,a,p),rr=!1;var x=t.memoizedState;o.state=x,xs(t,a,o,s),d=t.memoizedState,c!==a||x!==d||et.current||rr?(typeof v=="function"&&(to(t,n,v,a),d=t.memoizedState),(c=rr||dc(t,n,c,a,x,d,p))?(g||typeof o.UNSAFE_componentWillMount!="function"&&typeof o.componentWillMount!="function"||(typeof o.componentWillMount=="function"&&o.componentWillMount(),typeof o.UNSAFE_componentWillMount=="function"&&o.UNSAFE_componentWillMount()),typeof o.componentDidMount=="function"&&(t.flags|=4194308)):(typeof o.componentDidMount=="function"&&(t.flags|=4194308),t.memoizedProps=a,t.memoizedState=d),o.props=a,o.state=d,o.context=p,a=c):(typeof o.componentDidMount=="function"&&(t.flags|=4194308),a=!1)}else{o=t.stateNode,cu(e,t),c=t.memoizedProps,p=t.type===t.elementType?c:St(t.type,c),o.props=p,g=t.pendingProps,x=o.context,d=n.contextType,typeof d=="object"&&d!==null?d=yt(d):(d=tt(n)?Ir:Be.current,d=on(t,d));var k=n.getDerivedStateFromProps;(v=typeof k=="function"||typeof o.getSnapshotBeforeUpdate=="function")||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(c!==g||x!==d)&&uc(t,o,a,d),rr=!1,x=t.memoizedState,o.state=x,xs(t,a,o,s);var w=t.memoizedState;c!==g||x!==w||et.current||rr?(typeof k=="function"&&(to(t,n,k,a),w=t.memoizedState),(p=rr||dc(t,n,p,a,x,w,d)||!1)?(v||typeof o.UNSAFE_componentWillUpdate!="function"&&typeof o.componentWillUpdate!="function"||(typeof o.componentWillUpdate=="function"&&o.componentWillUpdate(a,w,d),typeof o.UNSAFE_componentWillUpdate=="function"&&o.UNSAFE_componentWillUpdate(a,w,d)),typeof o.componentDidUpdate=="function"&&(t.flags|=4),typeof o.getSnapshotBeforeUpdate=="function"&&(t.flags|=1024)):(typeof o.componentDidUpdate!="function"||c===e.memoizedProps&&x===e.memoizedState||(t.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||c===e.memoizedProps&&x===e.memoizedState||(t.flags|=1024),t.memoizedProps=a,t.memoizedState=w),o.props=a,o.state=w,o.context=d,a=p):(typeof o.componentDidUpdate!="function"||c===e.memoizedProps&&x===e.memoizedState||(t.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||c===e.memoizedProps&&x===e.memoizedState||(t.flags|=1024),a=!1)}return so(e,t,n,a,l,s)}function so(e,t,n,a,s,l){Fu(e,t);var o=(t.flags&128)!==0;if(!a&&!o)return s&&rc(t,n,!1),Xt(e,t,l);a=t.stateNode,_m.current=t;var c=o&&typeof n.getDerivedStateFromError!="function"?null:a.render();return t.flags|=1,e!==null&&o?(t.child=dn(t,e.child,null,l),t.child=dn(t,null,c,l)):Ge(e,t,c,l),t.memoizedState=a.state,s&&rc(t,n,!0),t.child}function Lu(e){var t=e.stateNode;t.pendingContext?tc(e,t.pendingContext,t.pendingContext!==t.context):t.context&&tc(e,t.context,!1),Xo(e,t.containerInfo)}function vc(e,t,n,a,s){return cn(),Wo(s),t.flags|=256,Ge(e,t,n,a),t.child}var lo={dehydrated:null,treeContext:null,retryLane:0};function oo(e){return{baseLanes:e,cachePool:null,transitions:null}}function Du(e,t,n){var a=t.pendingProps,s=Ne.current,l=!1,o=(t.flags&128)!==0,c;if((c=o)||(c=e!==null&&e.memoizedState===null?!1:(s&2)!==0),c?(l=!0,t.flags&=-129):(e===null||e.memoizedState!==null)&&(s|=1),be(Ne,s&1),e===null)return Zl(t),e=t.memoizedState,e!==null&&(e=e.dehydrated,e!==null)?(t.mode&1?e.data==="$!"?t.lanes=8:t.lanes=1073741824:t.lanes=1,null):(o=a.children,e=a.fallback,l?(a=t.mode,l=t.child,o={mode:"hidden",children:o},!(a&1)&&l!==null?(l.childLanes=0,l.pendingProps=o):l=Os(o,a,0,null),e=Pr(e,a,n,null),l.return=t,e.return=t,l.sibling=e,t.child=l,t.child.memoizedState=oo(n),t.memoizedState=lo,e):ai(t,o));if(s=e.memoizedState,s!==null&&(c=s.dehydrated,c!==null))return zm(e,t,o,a,c,s,n);if(l){l=a.fallback,o=t.mode,s=e.child,c=s.sibling;var d={mode:"hidden",children:a.children};return!(o&1)&&t.child!==s?(a=t.child,a.childLanes=0,a.pendingProps=d,t.deletions=null):(a=mr(s,d),a.subtreeFlags=s.subtreeFlags&14680064),c!==null?l=mr(c,l):(l=Pr(l,o,n,null),l.flags|=2),l.return=t,a.return=t,a.sibling=l,t.child=a,a=l,l=t.child,o=e.child.memoizedState,o=o===null?oo(n):{baseLanes:o.baseLanes|n,cachePool:null,transitions:o.transitions},l.memoizedState=o,l.childLanes=e.childLanes&~n,t.memoizedState=lo,a}return l=e.child,e=l.sibling,a=mr(l,{mode:"visible",children:a.children}),!(t.mode&1)&&(a.lanes=n),a.return=t,a.sibling=null,e!==null&&(n=t.deletions,n===null?(t.deletions=[e],t.flags|=16):n.push(e)),t.child=a,t.memoizedState=null,a}function ai(e,t){return t=Os({mode:"visible",children:t},e.mode,0,null),t.return=e,e.child=t}function Oa(e,t,n,a){return a!==null&&Wo(a),dn(t,e.child,null,n),e=ai(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function zm(e,t,n,a,s,l,o){if(n)return t.flags&256?(t.flags&=-257,a=hl(Error(B(422))),Oa(e,t,o,a)):t.memoizedState!==null?(t.child=e.child,t.flags|=128,null):(l=a.fallback,s=t.mode,a=Os({mode:"visible",children:a.children},s,0,null),l=Pr(l,s,o,null),l.flags|=2,a.return=t,l.return=t,a.sibling=l,t.child=a,t.mode&1&&dn(t,e.child,null,o),t.child.memoizedState=oo(o),t.memoizedState=lo,l);if(!(t.mode&1))return Oa(e,t,o,null);if(s.data==="$!"){if(a=s.nextSibling&&s.nextSibling.dataset,a)var c=a.dgst;return a=c,l=Error(B(419)),a=hl(l,a,void 0),Oa(e,t,o,a)}if(c=(o&e.childLanes)!==0,Ze||c){if(a=Fe,a!==null){switch(o&-o){case 4:s=2;break;case 16:s=8;break;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:s=32;break;case 536870912:s=268435456;break;default:s=0}s=s&(a.suspendedLanes|o)?0:s,s!==0&&s!==l.retryLane&&(l.retryLane=s,Yt(e,s),zt(a,e,s,-1))}return di(),a=hl(Error(B(421))),Oa(e,t,o,a)}return s.data==="$?"?(t.flags|=128,t.child=e.child,t=Um.bind(null,e),s._reactRetry=t,null):(e=l.treeContext,it=dr(s.nextSibling),ct=t,Se=!0,Ct=null,e!==null&&(mt[ht++]=Vt,mt[ht++]=Bt,mt[ht++]=Rr,Vt=e.id,Bt=e.overflow,Rr=t),t=ai(t,a.children),t.flags|=4096,t)}function yc(e,t,n){e.lanes|=t;var a=e.alternate;a!==null&&(a.lanes|=t),eo(e.return,t,n)}function xl(e,t,n,a,s){var l=e.memoizedState;l===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:a,tail:n,tailMode:s}:(l.isBackwards=t,l.rendering=null,l.renderingStartTime=0,l.last=a,l.tail=n,l.tailMode=s)}function Ou(e,t,n){var a=t.pendingProps,s=a.revealOrder,l=a.tail;if(Ge(e,t,a.children,n),a=Ne.current,a&2)a=a&1|2,t.flags|=128;else{if(e!==null&&e.flags&128)e:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&yc(e,n,t);else if(e.tag===19)yc(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break e;for(;e.sibling===null;){if(e.return===null||e.return===t)break e;e=e.return}e.sibling.return=e.return,e=e.sibling}a&=1}if(be(Ne,a),!(t.mode&1))t.memoizedState=null;else switch(s){case"forwards":for(n=t.child,s=null;n!==null;)e=n.alternate,e!==null&&gs(e)===null&&(s=n),n=n.sibling;n=s,n===null?(s=t.child,t.child=null):(s=n.sibling,n.sibling=null),xl(t,!1,s,n,l);break;case"backwards":for(n=null,s=t.child,t.child=null;s!==null;){if(e=s.alternate,e!==null&&gs(e)===null){t.child=s;break}e=s.sibling,s.sibling=n,n=s,s=e}xl(t,!0,n,null,l);break;case"together":xl(t,!1,null,null,void 0);break;default:t.memoizedState=null}return t.child}function Ka(e,t){!(t.mode&1)&&e!==null&&(e.alternate=null,t.alternate=null,t.flags|=2)}function Xt(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),Fr|=t.lanes,!(n&t.childLanes))return null;if(e!==null&&t.child!==e.child)throw Error(B(153));if(t.child!==null){for(e=t.child,n=mr(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=mr(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function Em(e,t,n){switch(t.tag){case 3:Lu(t),cn();break;case 5:du(t);break;case 1:tt(t.type)&&us(t);break;case 4:Xo(t,t.stateNode.containerInfo);break;case 10:var a=t.type._context,s=t.memoizedProps.value;be(ms,a._currentValue),a._currentValue=s;break;case 13:if(a=t.memoizedState,a!==null)return a.dehydrated!==null?(be(Ne,Ne.current&1),t.flags|=128,null):n&t.child.childLanes?Du(e,t,n):(be(Ne,Ne.current&1),e=Xt(e,t,n),e!==null?e.sibling:null);be(Ne,Ne.current&1);break;case 19:if(a=(n&t.childLanes)!==0,e.flags&128){if(a)return Ou(e,t,n);t.flags|=128}if(s=t.memoizedState,s!==null&&(s.rendering=null,s.tail=null,s.lastEffect=null),be(Ne,Ne.current),a)break;return null;case 22:case 23:return t.lanes=0,Mu(e,t,n)}return Xt(e,t,n)}var Au,io,$u,Uu;Au=function(e,t){for(var n=t.child;n!==null;){if(n.tag===5||n.tag===6)e.appendChild(n.stateNode);else if(n.tag!==4&&n.child!==null){n.child.return=n,n=n.child;continue}if(n===t)break;for(;n.sibling===null;){if(n.return===null||n.return===t)return;n=n.return}n.sibling.return=n.return,n=n.sibling}};io=function(){};$u=function(e,t,n,a){var s=e.memoizedProps;if(s!==a){e=t.stateNode,Er(Ot.current);var l=null;switch(n){case"input":s=Tl(e,s),a=Tl(e,a),l=[];break;case"select":s=_e({},s,{value:void 0}),a=_e({},a,{value:void 0}),l=[];break;case"textarea":s=Rl(e,s),a=Rl(e,a),l=[];break;default:typeof s.onClick!="function"&&typeof a.onClick=="function"&&(e.onclick=cs)}Fl(n,a);var o;n=null;for(p in s)if(!a.hasOwnProperty(p)&&s.hasOwnProperty(p)&&s[p]!=null)if(p==="style"){var c=s[p];for(o in c)c.hasOwnProperty(o)&&(n||(n={}),n[o]="")}else p!=="dangerouslySetInnerHTML"&&p!=="children"&&p!=="suppressContentEditableWarning"&&p!=="suppressHydrationWarning"&&p!=="autoFocus"&&(Qn.hasOwnProperty(p)?l||(l=[]):(l=l||[]).push(p,null));for(p in a){var d=a[p];if(c=s!=null?s[p]:void 0,a.hasOwnProperty(p)&&d!==c&&(d!=null||c!=null))if(p==="style")if(c){for(o in c)!c.hasOwnProperty(o)||d&&d.hasOwnProperty(o)||(n||(n={}),n[o]="");for(o in d)d.hasOwnProperty(o)&&c[o]!==d[o]&&(n||(n={}),n[o]=d[o])}else n||(l||(l=[]),l.push(p,n)),n=d;else p==="dangerouslySetInnerHTML"?(d=d?d.__html:void 0,c=c?c.__html:void 0,d!=null&&c!==d&&(l=l||[]).push(p,d)):p==="children"?typeof d!="string"&&typeof d!="number"||(l=l||[]).push(p,""+d):p!=="suppressContentEditableWarning"&&p!=="suppressHydrationWarning"&&(Qn.hasOwnProperty(p)?(d!=null&&p==="onScroll"&&we("scroll",e),l||c===d||(l=[])):(l=l||[]).push(p,d))}n&&(l=l||[]).push("style",n);var p=l;(t.updateQueue=p)&&(t.flags|=4)}};Uu=function(e,t,n,a){n!==a&&(t.flags|=4)};function In(e,t){if(!Se)switch(e.tailMode){case"hidden":t=e.tail;for(var n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null;break;case"collapsed":n=e.tail;for(var a=null;n!==null;)n.alternate!==null&&(a=n),n=n.sibling;a===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:a.sibling=null}}function Ue(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,a=0;if(t)for(var s=e.child;s!==null;)n|=s.lanes|s.childLanes,a|=s.subtreeFlags&14680064,a|=s.flags&14680064,s.return=e,s=s.sibling;else for(s=e.child;s!==null;)n|=s.lanes|s.childLanes,a|=s.subtreeFlags,a|=s.flags,s.return=e,s=s.sibling;return e.subtreeFlags|=a,e.childLanes=n,t}function Tm(e,t,n){var a=t.pendingProps;switch(Bo(t),t.tag){case 2:case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return Ue(t),null;case 1:return tt(t.type)&&ds(),Ue(t),null;case 3:return a=t.stateNode,un(),ke(et),ke(Be),Jo(),a.pendingContext&&(a.context=a.pendingContext,a.pendingContext=null),(e===null||e.child===null)&&(La(t)?t.flags|=4:e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,Ct!==null&&(go(Ct),Ct=null))),io(e,t),Ue(t),null;case 5:Ko(t);var s=Er(sa.current);if(n=t.type,e!==null&&t.stateNode!=null)$u(e,t,n,a,s),e.ref!==t.ref&&(t.flags|=512,t.flags|=2097152);else{if(!a){if(t.stateNode===null)throw Error(B(166));return Ue(t),null}if(e=Er(Ot.current),La(t)){a=t.stateNode,n=t.type;var l=t.memoizedProps;switch(a[Lt]=t,a[na]=l,e=(t.mode&1)!==0,n){case"dialog":we("cancel",a),we("close",a);break;case"iframe":case"object":case"embed":we("load",a);break;case"video":case"audio":for(s=0;s<Dn.length;s++)we(Dn[s],a);break;case"source":we("error",a);break;case"img":case"image":case"link":we("error",a),we("load",a);break;case"details":we("toggle",a);break;case"input":zi(a,l),we("invalid",a);break;case"select":a._wrapperState={wasMultiple:!!l.multiple},we("invalid",a);break;case"textarea":Ti(a,l),we("invalid",a)}Fl(n,l),s=null;for(var o in l)if(l.hasOwnProperty(o)){var c=l[o];o==="children"?typeof c=="string"?a.textContent!==c&&(l.suppressHydrationWarning!==!0&&Fa(a.textContent,c,e),s=["children",c]):typeof c=="number"&&a.textContent!==""+c&&(l.suppressHydrationWarning!==!0&&Fa(a.textContent,c,e),s=["children",""+c]):Qn.hasOwnProperty(o)&&c!=null&&o==="onScroll"&&we("scroll",a)}switch(n){case"input":_a(a),Ei(a,l,!0);break;case"textarea":_a(a),Pi(a);break;case"select":case"option":break;default:typeof l.onClick=="function"&&(a.onclick=cs)}a=s,t.updateQueue=a,a!==null&&(t.flags|=4)}else{o=s.nodeType===9?s:s.ownerDocument,e==="http://www.w3.org/1999/xhtml"&&(e=hd(n)),e==="http://www.w3.org/1999/xhtml"?n==="script"?(e=o.createElement("div"),e.innerHTML="<script><\/script>",e=e.removeChild(e.firstChild)):typeof a.is=="string"?e=o.createElement(n,{is:a.is}):(e=o.createElement(n),n==="select"&&(o=e,a.multiple?o.multiple=!0:a.size&&(o.size=a.size))):e=o.createElementNS(e,n),e[Lt]=t,e[na]=a,Au(e,t,!1,!1),t.stateNode=e;e:{switch(o=Ll(n,a),n){case"dialog":we("cancel",e),we("close",e),s=a;break;case"iframe":case"object":case"embed":we("load",e),s=a;break;case"video":case"audio":for(s=0;s<Dn.length;s++)we(Dn[s],e);s=a;break;case"source":we("error",e),s=a;break;case"img":case"image":case"link":we("error",e),we("load",e),s=a;break;case"details":we("toggle",e),s=a;break;case"input":zi(e,a),s=Tl(e,a),we("invalid",e);break;case"option":s=a;break;case"select":e._wrapperState={wasMultiple:!!a.multiple},s=_e({},a,{value:void 0}),we("invalid",e);break;case"textarea":Ti(e,a),s=Rl(e,a),we("invalid",e);break;default:s=a}Fl(n,s),c=s;for(l in c)if(c.hasOwnProperty(l)){var d=c[l];l==="style"?vd(e,d):l==="dangerouslySetInnerHTML"?(d=d?d.__html:void 0,d!=null&&xd(e,d)):l==="children"?typeof d=="string"?(n!=="textarea"||d!=="")&&qn(e,d):typeof d=="number"&&qn(e,""+d):l!=="suppressContentEditableWarning"&&l!=="suppressHydrationWarning"&&l!=="autoFocus"&&(Qn.hasOwnProperty(l)?d!=null&&l==="onScroll"&&we("scroll",e):d!=null&&zo(e,l,d,o))}switch(n){case"input":_a(e),Ei(e,a,!1);break;case"textarea":_a(e),Pi(e);break;case"option":a.value!=null&&e.setAttribute("value",""+hr(a.value));break;case"select":e.multiple=!!a.multiple,l=a.value,l!=null?en(e,!!a.multiple,l,!1):a.defaultValue!=null&&en(e,!!a.multiple,a.defaultValue,!0);break;default:typeof s.onClick=="function"&&(e.onclick=cs)}switch(n){case"button":case"input":case"select":case"textarea":a=!!a.autoFocus;break e;case"img":a=!0;break e;default:a=!1}}a&&(t.flags|=4)}t.ref!==null&&(t.flags|=512,t.flags|=2097152)}return Ue(t),null;case 6:if(e&&t.stateNode!=null)Uu(e,t,e.memoizedProps,a);else{if(typeof a!="string"&&t.stateNode===null)throw Error(B(166));if(n=Er(sa.current),Er(Ot.current),La(t)){if(a=t.stateNode,n=t.memoizedProps,a[Lt]=t,(l=a.nodeValue!==n)&&(e=ct,e!==null))switch(e.tag){case 3:Fa(a.nodeValue,n,(e.mode&1)!==0);break;case 5:e.memoizedProps.suppressHydrationWarning!==!0&&Fa(a.nodeValue,n,(e.mode&1)!==0)}l&&(t.flags|=4)}else a=(n.nodeType===9?n:n.ownerDocument).createTextNode(a),a[Lt]=t,t.stateNode=a}return Ue(t),null;case 13:if(ke(Ne),a=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(Se&&it!==null&&t.mode&1&&!(t.flags&128))su(),cn(),t.flags|=98560,l=!1;else if(l=La(t),a!==null&&a.dehydrated!==null){if(e===null){if(!l)throw Error(B(318));if(l=t.memoizedState,l=l!==null?l.dehydrated:null,!l)throw Error(B(317));l[Lt]=t}else cn(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;Ue(t),l=!1}else Ct!==null&&(go(Ct),Ct=null),l=!0;if(!l)return t.flags&65536?t:null}return t.flags&128?(t.lanes=n,t):(a=a!==null,a!==(e!==null&&e.memoizedState!==null)&&a&&(t.child.flags|=8192,t.mode&1&&(e===null||Ne.current&1?Re===0&&(Re=3):di())),t.updateQueue!==null&&(t.flags|=4),Ue(t),null);case 4:return un(),io(e,t),e===null&&ta(t.stateNode.containerInfo),Ue(t),null;case 10:return Qo(t.type._context),Ue(t),null;case 17:return tt(t.type)&&ds(),Ue(t),null;case 19:if(ke(Ne),l=t.memoizedState,l===null)return Ue(t),null;if(a=(t.flags&128)!==0,o=l.rendering,o===null)if(a)In(l,!1);else{if(Re!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(o=gs(e),o!==null){for(t.flags|=128,In(l,!1),a=o.updateQueue,a!==null&&(t.updateQueue=a,t.flags|=4),t.subtreeFlags=0,a=n,n=t.child;n!==null;)l=n,e=a,l.flags&=14680066,o=l.alternate,o===null?(l.childLanes=0,l.lanes=e,l.child=null,l.subtreeFlags=0,l.memoizedProps=null,l.memoizedState=null,l.updateQueue=null,l.dependencies=null,l.stateNode=null):(l.childLanes=o.childLanes,l.lanes=o.lanes,l.child=o.child,l.subtreeFlags=0,l.deletions=null,l.memoizedProps=o.memoizedProps,l.memoizedState=o.memoizedState,l.updateQueue=o.updateQueue,l.type=o.type,e=o.dependencies,l.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),n=n.sibling;return be(Ne,Ne.current&1|2),t.child}e=e.sibling}l.tail!==null&&Ee()>fn&&(t.flags|=128,a=!0,In(l,!1),t.lanes=4194304)}else{if(!a)if(e=gs(o),e!==null){if(t.flags|=128,a=!0,n=e.updateQueue,n!==null&&(t.updateQueue=n,t.flags|=4),In(l,!0),l.tail===null&&l.tailMode==="hidden"&&!o.alternate&&!Se)return Ue(t),null}else 2*Ee()-l.renderingStartTime>fn&&n!==1073741824&&(t.flags|=128,a=!0,In(l,!1),t.lanes=4194304);l.isBackwards?(o.sibling=t.child,t.child=o):(n=l.last,n!==null?n.sibling=o:t.child=o,l.last=o)}return l.tail!==null?(t=l.tail,l.rendering=t,l.tail=t.sibling,l.renderingStartTime=Ee(),t.sibling=null,n=Ne.current,be(Ne,a?n&1|2:n&1),t):(Ue(t),null);case 22:case 23:return ci(),a=t.memoizedState!==null,e!==null&&e.memoizedState!==null!==a&&(t.flags|=8192),a&&t.mode&1?ot&1073741824&&(Ue(t),t.subtreeFlags&6&&(t.flags|=8192)):Ue(t),null;case 24:return null;case 25:return null}throw Error(B(156,t.tag))}function Pm(e,t){switch(Bo(t),t.tag){case 1:return tt(t.type)&&ds(),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return un(),ke(et),ke(Be),Jo(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 5:return Ko(t),null;case 13:if(ke(Ne),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(B(340));cn()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return ke(Ne),null;case 4:return un(),null;case 10:return Qo(t.type._context),null;case 22:case 23:return ci(),null;case 24:return null;default:return null}}var Aa=!1,Ve=!1,Im=typeof WeakSet=="function"?WeakSet:Set,se=null;function Jr(e,t){var n=e.ref;if(n!==null)if(typeof n=="function")try{n(null)}catch(a){ze(e,t,a)}else n.current=null}function co(e,t,n){try{n()}catch(a){ze(e,t,a)}}var jc=!1;function Rm(e,t){if(Gl=ls,e=Gd(),Uo(e)){if("selectionStart"in e)var n={start:e.selectionStart,end:e.selectionEnd};else e:{n=(n=e.ownerDocument)&&n.defaultView||window;var a=n.getSelection&&n.getSelection();if(a&&a.rangeCount!==0){n=a.anchorNode;var s=a.anchorOffset,l=a.focusNode;a=a.focusOffset;try{n.nodeType,l.nodeType}catch{n=null;break e}var o=0,c=-1,d=-1,p=0,v=0,g=e,x=null;t:for(;;){for(var k;g!==n||s!==0&&g.nodeType!==3||(c=o+s),g!==l||a!==0&&g.nodeType!==3||(d=o+a),g.nodeType===3&&(o+=g.nodeValue.length),(k=g.firstChild)!==null;)x=g,g=k;for(;;){if(g===e)break t;if(x===n&&++p===s&&(c=o),x===l&&++v===a&&(d=o),(k=g.nextSibling)!==null)break;g=x,x=g.parentNode}g=k}n=c===-1||d===-1?null:{start:c,end:d}}else n=null}n=n||{start:0,end:0}}else n=null;for(Ql={focusedElem:e,selectionRange:n},ls=!1,se=t;se!==null;)if(t=se,e=t.child,(t.subtreeFlags&1028)!==0&&e!==null)e.return=t,se=e;else for(;se!==null;){t=se;try{var w=t.alternate;if(t.flags&1024)switch(t.tag){case 0:case 11:case 15:break;case 1:if(w!==null){var z=w.memoizedProps,F=w.memoizedState,f=t.stateNode,u=f.getSnapshotBeforeUpdate(t.elementType===t.type?z:St(t.type,z),F);f.__reactInternalSnapshotBeforeUpdate=u}break;case 3:var h=t.stateNode.containerInfo;h.nodeType===1?h.textContent="":h.nodeType===9&&h.documentElement&&h.removeChild(h.documentElement);break;case 5:case 6:case 4:case 17:break;default:throw Error(B(163))}}catch(y){ze(t,t.return,y)}if(e=t.sibling,e!==null){e.return=t.return,se=e;break}se=t.return}return w=jc,jc=!1,w}function Wn(e,t,n){var a=t.updateQueue;if(a=a!==null?a.lastEffect:null,a!==null){var s=a=a.next;do{if((s.tag&e)===e){var l=s.destroy;s.destroy=void 0,l!==void 0&&co(t,n,l)}s=s.next}while(s!==a)}}function Ls(e,t){if(t=t.updateQueue,t=t!==null?t.lastEffect:null,t!==null){var n=t=t.next;do{if((n.tag&e)===e){var a=n.create;n.destroy=a()}n=n.next}while(n!==t)}}function uo(e){var t=e.ref;if(t!==null){var n=e.stateNode;switch(e.tag){case 5:e=n;break;default:e=n}typeof t=="function"?t(e):t.current=e}}function Vu(e){var t=e.alternate;t!==null&&(e.alternate=null,Vu(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&(delete t[Lt],delete t[na],delete t[Xl],delete t[hm],delete t[xm])),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}function Bu(e){return e.tag===5||e.tag===3||e.tag===4}function bc(e){e:for(;;){for(;e.sibling===null;){if(e.return===null||Bu(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.flags&2||e.child===null||e.tag===4)continue e;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function po(e,t,n){var a=e.tag;if(a===5||a===6)e=e.stateNode,t?n.nodeType===8?n.parentNode.insertBefore(e,t):n.insertBefore(e,t):(n.nodeType===8?(t=n.parentNode,t.insertBefore(e,n)):(t=n,t.appendChild(e)),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=cs));else if(a!==4&&(e=e.child,e!==null))for(po(e,t,n),e=e.sibling;e!==null;)po(e,t,n),e=e.sibling}function fo(e,t,n){var a=e.tag;if(a===5||a===6)e=e.stateNode,t?n.insertBefore(e,t):n.appendChild(e);else if(a!==4&&(e=e.child,e!==null))for(fo(e,t,n),e=e.sibling;e!==null;)fo(e,t,n),e=e.sibling}var Le=null,Nt=!1;function er(e,t,n){for(n=n.child;n!==null;)Wu(e,t,n),n=n.sibling}function Wu(e,t,n){if(Dt&&typeof Dt.onCommitFiberUnmount=="function")try{Dt.onCommitFiberUnmount(zs,n)}catch{}switch(n.tag){case 5:Ve||Jr(n,t);case 6:var a=Le,s=Nt;Le=null,er(e,t,n),Le=a,Nt=s,Le!==null&&(Nt?(e=Le,n=n.stateNode,e.nodeType===8?e.parentNode.removeChild(n):e.removeChild(n)):Le.removeChild(n.stateNode));break;case 18:Le!==null&&(Nt?(e=Le,n=n.stateNode,e.nodeType===8?cl(e.parentNode,n):e.nodeType===1&&cl(e,n),Jn(e)):cl(Le,n.stateNode));break;case 4:a=Le,s=Nt,Le=n.stateNode.containerInfo,Nt=!0,er(e,t,n),Le=a,Nt=s;break;case 0:case 11:case 14:case 15:if(!Ve&&(a=n.updateQueue,a!==null&&(a=a.lastEffect,a!==null))){s=a=a.next;do{var l=s,o=l.destroy;l=l.tag,o!==void 0&&(l&2||l&4)&&co(n,t,o),s=s.next}while(s!==a)}er(e,t,n);break;case 1:if(!Ve&&(Jr(n,t),a=n.stateNode,typeof a.componentWillUnmount=="function"))try{a.props=n.memoizedProps,a.state=n.memoizedState,a.componentWillUnmount()}catch(c){ze(n,t,c)}er(e,t,n);break;case 21:er(e,t,n);break;case 22:n.mode&1?(Ve=(a=Ve)||n.memoizedState!==null,er(e,t,n),Ve=a):er(e,t,n);break;default:er(e,t,n)}}function wc(e){var t=e.updateQueue;if(t!==null){e.updateQueue=null;var n=e.stateNode;n===null&&(n=e.stateNode=new Im),t.forEach(function(a){var s=Vm.bind(null,e,a);n.has(a)||(n.add(a),a.then(s,s))})}}function kt(e,t){var n=t.deletions;if(n!==null)for(var a=0;a<n.length;a++){var s=n[a];try{var l=e,o=t,c=o;e:for(;c!==null;){switch(c.tag){case 5:Le=c.stateNode,Nt=!1;break e;case 3:Le=c.stateNode.containerInfo,Nt=!0;break e;case 4:Le=c.stateNode.containerInfo,Nt=!0;break e}c=c.return}if(Le===null)throw Error(B(160));Wu(l,o,s),Le=null,Nt=!1;var d=s.alternate;d!==null&&(d.return=null),s.return=null}catch(p){ze(s,t,p)}}if(t.subtreeFlags&12854)for(t=t.child;t!==null;)Hu(t,e),t=t.sibling}function Hu(e,t){var n=e.alternate,a=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(kt(t,e),Rt(e),a&4){try{Wn(3,e,e.return),Ls(3,e)}catch(z){ze(e,e.return,z)}try{Wn(5,e,e.return)}catch(z){ze(e,e.return,z)}}break;case 1:kt(t,e),Rt(e),a&512&&n!==null&&Jr(n,n.return);break;case 5:if(kt(t,e),Rt(e),a&512&&n!==null&&Jr(n,n.return),e.flags&32){var s=e.stateNode;try{qn(s,"")}catch(z){ze(e,e.return,z)}}if(a&4&&(s=e.stateNode,s!=null)){var l=e.memoizedProps,o=n!==null?n.memoizedProps:l,c=e.type,d=e.updateQueue;if(e.updateQueue=null,d!==null)try{c==="input"&&l.type==="radio"&&l.name!=null&&fd(s,l),Ll(c,o);var p=Ll(c,l);for(o=0;o<d.length;o+=2){var v=d[o],g=d[o+1];v==="style"?vd(s,g):v==="dangerouslySetInnerHTML"?xd(s,g):v==="children"?qn(s,g):zo(s,v,g,p)}switch(c){case"input":Pl(s,l);break;case"textarea":md(s,l);break;case"select":var x=s._wrapperState.wasMultiple;s._wrapperState.wasMultiple=!!l.multiple;var k=l.value;k!=null?en(s,!!l.multiple,k,!1):x!==!!l.multiple&&(l.defaultValue!=null?en(s,!!l.multiple,l.defaultValue,!0):en(s,!!l.multiple,l.multiple?[]:"",!1))}s[na]=l}catch(z){ze(e,e.return,z)}}break;case 6:if(kt(t,e),Rt(e),a&4){if(e.stateNode===null)throw Error(B(162));s=e.stateNode,l=e.memoizedProps;try{s.nodeValue=l}catch(z){ze(e,e.return,z)}}break;case 3:if(kt(t,e),Rt(e),a&4&&n!==null&&n.memoizedState.isDehydrated)try{Jn(t.containerInfo)}catch(z){ze(e,e.return,z)}break;case 4:kt(t,e),Rt(e);break;case 13:kt(t,e),Rt(e),s=e.child,s.flags&8192&&(l=s.memoizedState!==null,s.stateNode.isHidden=l,!l||s.alternate!==null&&s.alternate.memoizedState!==null||(oi=Ee())),a&4&&wc(e);break;case 22:if(v=n!==null&&n.memoizedState!==null,e.mode&1?(Ve=(p=Ve)||v,kt(t,e),Ve=p):kt(t,e),Rt(e),a&8192){if(p=e.memoizedState!==null,(e.stateNode.isHidden=p)&&!v&&e.mode&1)for(se=e,v=e.child;v!==null;){for(g=se=v;se!==null;){switch(x=se,k=x.child,x.tag){case 0:case 11:case 14:case 15:Wn(4,x,x.return);break;case 1:Jr(x,x.return);var w=x.stateNode;if(typeof w.componentWillUnmount=="function"){a=x,n=x.return;try{t=a,w.props=t.memoizedProps,w.state=t.memoizedState,w.componentWillUnmount()}catch(z){ze(a,n,z)}}break;case 5:Jr(x,x.return);break;case 22:if(x.memoizedState!==null){Sc(g);continue}}k!==null?(k.return=x,se=k):Sc(g)}v=v.sibling}e:for(v=null,g=e;;){if(g.tag===5){if(v===null){v=g;try{s=g.stateNode,p?(l=s.style,typeof l.setProperty=="function"?l.setProperty("display","none","important"):l.display="none"):(c=g.stateNode,d=g.memoizedProps.style,o=d!=null&&d.hasOwnProperty("display")?d.display:null,c.style.display=gd("display",o))}catch(z){ze(e,e.return,z)}}}else if(g.tag===6){if(v===null)try{g.stateNode.nodeValue=p?"":g.memoizedProps}catch(z){ze(e,e.return,z)}}else if((g.tag!==22&&g.tag!==23||g.memoizedState===null||g===e)&&g.child!==null){g.child.return=g,g=g.child;continue}if(g===e)break e;for(;g.sibling===null;){if(g.return===null||g.return===e)break e;v===g&&(v=null),g=g.return}v===g&&(v=null),g.sibling.return=g.return,g=g.sibling}}break;case 19:kt(t,e),Rt(e),a&4&&wc(e);break;case 21:break;default:kt(t,e),Rt(e)}}function Rt(e){var t=e.flags;if(t&2){try{e:{for(var n=e.return;n!==null;){if(Bu(n)){var a=n;break e}n=n.return}throw Error(B(160))}switch(a.tag){case 5:var s=a.stateNode;a.flags&32&&(qn(s,""),a.flags&=-33);var l=bc(e);fo(e,l,s);break;case 3:case 4:var o=a.stateNode.containerInfo,c=bc(e);po(e,c,o);break;default:throw Error(B(161))}}catch(d){ze(e,e.return,d)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function Mm(e,t,n){se=e,Gu(e)}function Gu(e,t,n){for(var a=(e.mode&1)!==0;se!==null;){var s=se,l=s.child;if(s.tag===22&&a){var o=s.memoizedState!==null||Aa;if(!o){var c=s.alternate,d=c!==null&&c.memoizedState!==null||Ve;c=Aa;var p=Ve;if(Aa=o,(Ve=d)&&!p)for(se=s;se!==null;)o=se,d=o.child,o.tag===22&&o.memoizedState!==null?Nc(s):d!==null?(d.return=o,se=d):Nc(s);for(;l!==null;)se=l,Gu(l),l=l.sibling;se=s,Aa=c,Ve=p}kc(e)}else s.subtreeFlags&8772&&l!==null?(l.return=s,se=l):kc(e)}}function kc(e){for(;se!==null;){var t=se;if(t.flags&8772){var n=t.alternate;try{if(t.flags&8772)switch(t.tag){case 0:case 11:case 15:Ve||Ls(5,t);break;case 1:var a=t.stateNode;if(t.flags&4&&!Ve)if(n===null)a.componentDidMount();else{var s=t.elementType===t.type?n.memoizedProps:St(t.type,n.memoizedProps);a.componentDidUpdate(s,n.memoizedState,a.__reactInternalSnapshotBeforeUpdate)}var l=t.updateQueue;l!==null&&oc(t,l,a);break;case 3:var o=t.updateQueue;if(o!==null){if(n=null,t.child!==null)switch(t.child.tag){case 5:n=t.child.stateNode;break;case 1:n=t.child.stateNode}oc(t,o,n)}break;case 5:var c=t.stateNode;if(n===null&&t.flags&4){n=c;var d=t.memoizedProps;switch(t.type){case"button":case"input":case"select":case"textarea":d.autoFocus&&n.focus();break;case"img":d.src&&(n.src=d.src)}}break;case 6:break;case 4:break;case 12:break;case 13:if(t.memoizedState===null){var p=t.alternate;if(p!==null){var v=p.memoizedState;if(v!==null){var g=v.dehydrated;g!==null&&Jn(g)}}}break;case 19:case 17:case 21:case 22:case 23:case 25:break;default:throw Error(B(163))}Ve||t.flags&512&&uo(t)}catch(x){ze(t,t.return,x)}}if(t===e){se=null;break}if(n=t.sibling,n!==null){n.return=t.return,se=n;break}se=t.return}}function Sc(e){for(;se!==null;){var t=se;if(t===e){se=null;break}var n=t.sibling;if(n!==null){n.return=t.return,se=n;break}se=t.return}}function Nc(e){for(;se!==null;){var t=se;try{switch(t.tag){case 0:case 11:case 15:var n=t.return;try{Ls(4,t)}catch(d){ze(t,n,d)}break;case 1:var a=t.stateNode;if(typeof a.componentDidMount=="function"){var s=t.return;try{a.componentDidMount()}catch(d){ze(t,s,d)}}var l=t.return;try{uo(t)}catch(d){ze(t,l,d)}break;case 5:var o=t.return;try{uo(t)}catch(d){ze(t,o,d)}}}catch(d){ze(t,t.return,d)}if(t===e){se=null;break}var c=t.sibling;if(c!==null){c.return=t.return,se=c;break}se=t.return}}var Fm=Math.ceil,js=Jt.ReactCurrentDispatcher,si=Jt.ReactCurrentOwner,gt=Jt.ReactCurrentBatchConfig,ge=0,Fe=null,Pe=null,De=0,ot=0,Zr=br(0),Re=0,ca=null,Fr=0,Ds=0,li=0,Hn=null,Je=null,oi=0,fn=1/0,$t=null,bs=!1,mo=null,pr=null,$a=!1,lr=null,ws=0,Gn=0,ho=null,Ja=-1,Za=0;function qe(){return ge&6?Ee():Ja!==-1?Ja:Ja=Ee()}function fr(e){return e.mode&1?ge&2&&De!==0?De&-De:vm.transition!==null?(Za===0&&(Za=Td()),Za):(e=je,e!==0||(e=window.event,e=e===void 0?16:Dd(e.type)),e):1}function zt(e,t,n,a){if(50<Gn)throw Gn=0,ho=null,Error(B(185));ha(e,n,a),(!(ge&2)||e!==Fe)&&(e===Fe&&(!(ge&2)&&(Ds|=n),Re===4&&ar(e,De)),rt(e,a),n===1&&ge===0&&!(t.mode&1)&&(fn=Ee()+500,Rs&&wr()))}function rt(e,t){var n=e.callbackNode;vf(e,t);var a=ss(e,e===Fe?De:0);if(a===0)n!==null&&Mi(n),e.callbackNode=null,e.callbackPriority=0;else if(t=a&-a,e.callbackPriority!==t){if(n!=null&&Mi(n),t===1)e.tag===0?gm(Cc.bind(null,e)):ru(Cc.bind(null,e)),fm(function(){!(ge&6)&&wr()}),n=null;else{switch(Pd(a)){case 1:n=Ro;break;case 4:n=zd;break;case 16:n=as;break;case 536870912:n=Ed;break;default:n=as}n=ep(n,Qu.bind(null,e))}e.callbackPriority=t,e.callbackNode=n}}function Qu(e,t){if(Ja=-1,Za=0,ge&6)throw Error(B(327));var n=e.callbackNode;if(sn()&&e.callbackNode!==n)return null;var a=ss(e,e===Fe?De:0);if(a===0)return null;if(a&30||a&e.expiredLanes||t)t=ks(e,a);else{t=a;var s=ge;ge|=2;var l=Yu();(Fe!==e||De!==t)&&($t=null,fn=Ee()+500,Tr(e,t));do try{Om();break}catch(c){qu(e,c)}while(!0);Go(),js.current=l,ge=s,Pe!==null?t=0:(Fe=null,De=0,t=Re)}if(t!==0){if(t===2&&(s=Ul(e),s!==0&&(a=s,t=xo(e,s))),t===1)throw n=ca,Tr(e,0),ar(e,a),rt(e,Ee()),n;if(t===6)ar(e,a);else{if(s=e.current.alternate,!(a&30)&&!Lm(s)&&(t=ks(e,a),t===2&&(l=Ul(e),l!==0&&(a=l,t=xo(e,l))),t===1))throw n=ca,Tr(e,0),ar(e,a),rt(e,Ee()),n;switch(e.finishedWork=s,e.finishedLanes=a,t){case 0:case 1:throw Error(B(345));case 2:Cr(e,Je,$t);break;case 3:if(ar(e,a),(a&130023424)===a&&(t=oi+500-Ee(),10<t)){if(ss(e,0)!==0)break;if(s=e.suspendedLanes,(s&a)!==a){qe(),e.pingedLanes|=e.suspendedLanes&s;break}e.timeoutHandle=Yl(Cr.bind(null,e,Je,$t),t);break}Cr(e,Je,$t);break;case 4:if(ar(e,a),(a&4194240)===a)break;for(t=e.eventTimes,s=-1;0<a;){var o=31-_t(a);l=1<<o,o=t[o],o>s&&(s=o),a&=~l}if(a=s,a=Ee()-a,a=(120>a?120:480>a?480:1080>a?1080:1920>a?1920:3e3>a?3e3:4320>a?4320:1960*Fm(a/1960))-a,10<a){e.timeoutHandle=Yl(Cr.bind(null,e,Je,$t),a);break}Cr(e,Je,$t);break;case 5:Cr(e,Je,$t);break;default:throw Error(B(329))}}}return rt(e,Ee()),e.callbackNode===n?Qu.bind(null,e):null}function xo(e,t){var n=Hn;return e.current.memoizedState.isDehydrated&&(Tr(e,t).flags|=256),e=ks(e,t),e!==2&&(t=Je,Je=n,t!==null&&go(t)),e}function go(e){Je===null?Je=e:Je.push.apply(Je,e)}function Lm(e){for(var t=e;;){if(t.flags&16384){var n=t.updateQueue;if(n!==null&&(n=n.stores,n!==null))for(var a=0;a<n.length;a++){var s=n[a],l=s.getSnapshot;s=s.value;try{if(!Et(l(),s))return!1}catch{return!1}}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function ar(e,t){for(t&=~li,t&=~Ds,e.suspendedLanes|=t,e.pingedLanes&=~t,e=e.expirationTimes;0<t;){var n=31-_t(t),a=1<<n;e[n]=-1,t&=~a}}function Cc(e){if(ge&6)throw Error(B(327));sn();var t=ss(e,0);if(!(t&1))return rt(e,Ee()),null;var n=ks(e,t);if(e.tag!==0&&n===2){var a=Ul(e);a!==0&&(t=a,n=xo(e,a))}if(n===1)throw n=ca,Tr(e,0),ar(e,t),rt(e,Ee()),n;if(n===6)throw Error(B(345));return e.finishedWork=e.current.alternate,e.finishedLanes=t,Cr(e,Je,$t),rt(e,Ee()),null}function ii(e,t){var n=ge;ge|=1;try{return e(t)}finally{ge=n,ge===0&&(fn=Ee()+500,Rs&&wr())}}function Lr(e){lr!==null&&lr.tag===0&&!(ge&6)&&sn();var t=ge;ge|=1;var n=gt.transition,a=je;try{if(gt.transition=null,je=1,e)return e()}finally{je=a,gt.transition=n,ge=t,!(ge&6)&&wr()}}function ci(){ot=Zr.current,ke(Zr)}function Tr(e,t){e.finishedWork=null,e.finishedLanes=0;var n=e.timeoutHandle;if(n!==-1&&(e.timeoutHandle=-1,pm(n)),Pe!==null)for(n=Pe.return;n!==null;){var a=n;switch(Bo(a),a.tag){case 1:a=a.type.childContextTypes,a!=null&&ds();break;case 3:un(),ke(et),ke(Be),Jo();break;case 5:Ko(a);break;case 4:un();break;case 13:ke(Ne);break;case 19:ke(Ne);break;case 10:Qo(a.type._context);break;case 22:case 23:ci()}n=n.return}if(Fe=e,Pe=e=mr(e.current,null),De=ot=t,Re=0,ca=null,li=Ds=Fr=0,Je=Hn=null,zr!==null){for(t=0;t<zr.length;t++)if(n=zr[t],a=n.interleaved,a!==null){n.interleaved=null;var s=a.next,l=n.pending;if(l!==null){var o=l.next;l.next=s,a.next=o}n.pending=a}zr=null}return e}function qu(e,t){do{var n=Pe;try{if(Go(),Ya.current=ys,vs){for(var a=Ce.memoizedState;a!==null;){var s=a.queue;s!==null&&(s.pending=null),a=a.next}vs=!1}if(Mr=0,Me=Ie=Ce=null,Bn=!1,la=0,si.current=null,n===null||n.return===null){Re=1,ca=t,Pe=null;break}e:{var l=e,o=n.return,c=n,d=t;if(t=De,c.flags|=32768,d!==null&&typeof d=="object"&&typeof d.then=="function"){var p=d,v=c,g=v.tag;if(!(v.mode&1)&&(g===0||g===11||g===15)){var x=v.alternate;x?(v.updateQueue=x.updateQueue,v.memoizedState=x.memoizedState,v.lanes=x.lanes):(v.updateQueue=null,v.memoizedState=null)}var k=fc(o);if(k!==null){k.flags&=-257,mc(k,o,c,l,t),k.mode&1&&pc(l,p,t),t=k,d=p;var w=t.updateQueue;if(w===null){var z=new Set;z.add(d),t.updateQueue=z}else w.add(d);break e}else{if(!(t&1)){pc(l,p,t),di();break e}d=Error(B(426))}}else if(Se&&c.mode&1){var F=fc(o);if(F!==null){!(F.flags&65536)&&(F.flags|=256),mc(F,o,c,l,t),Wo(pn(d,c));break e}}l=d=pn(d,c),Re!==4&&(Re=2),Hn===null?Hn=[l]:Hn.push(l),l=o;do{switch(l.tag){case 3:l.flags|=65536,t&=-t,l.lanes|=t;var f=Pu(l,d,t);lc(l,f);break e;case 1:c=d;var u=l.type,h=l.stateNode;if(!(l.flags&128)&&(typeof u.getDerivedStateFromError=="function"||h!==null&&typeof h.componentDidCatch=="function"&&(pr===null||!pr.has(h)))){l.flags|=65536,t&=-t,l.lanes|=t;var y=Iu(l,c,t);lc(l,y);break e}}l=l.return}while(l!==null)}Ku(n)}catch(j){t=j,Pe===n&&n!==null&&(Pe=n=n.return);continue}break}while(!0)}function Yu(){var e=js.current;return js.current=ys,e===null?ys:e}function di(){(Re===0||Re===3||Re===2)&&(Re=4),Fe===null||!(Fr&268435455)&&!(Ds&268435455)||ar(Fe,De)}function ks(e,t){var n=ge;ge|=2;var a=Yu();(Fe!==e||De!==t)&&($t=null,Tr(e,t));do try{Dm();break}catch(s){qu(e,s)}while(!0);if(Go(),ge=n,js.current=a,Pe!==null)throw Error(B(261));return Fe=null,De=0,Re}function Dm(){for(;Pe!==null;)Xu(Pe)}function Om(){for(;Pe!==null&&!cf();)Xu(Pe)}function Xu(e){var t=Zu(e.alternate,e,ot);e.memoizedProps=e.pendingProps,t===null?Ku(e):Pe=t,si.current=null}function Ku(e){var t=e;do{var n=t.alternate;if(e=t.return,t.flags&32768){if(n=Pm(n,t),n!==null){n.flags&=32767,Pe=n;return}if(e!==null)e.flags|=32768,e.subtreeFlags=0,e.deletions=null;else{Re=6,Pe=null;return}}else if(n=Tm(n,t,ot),n!==null){Pe=n;return}if(t=t.sibling,t!==null){Pe=t;return}Pe=t=e}while(t!==null);Re===0&&(Re=5)}function Cr(e,t,n){var a=je,s=gt.transition;try{gt.transition=null,je=1,Am(e,t,n,a)}finally{gt.transition=s,je=a}return null}function Am(e,t,n,a){do sn();while(lr!==null);if(ge&6)throw Error(B(327));n=e.finishedWork;var s=e.finishedLanes;if(n===null)return null;if(e.finishedWork=null,e.finishedLanes=0,n===e.current)throw Error(B(177));e.callbackNode=null,e.callbackPriority=0;var l=n.lanes|n.childLanes;if(yf(e,l),e===Fe&&(Pe=Fe=null,De=0),!(n.subtreeFlags&2064)&&!(n.flags&2064)||$a||($a=!0,ep(as,function(){return sn(),null})),l=(n.flags&15990)!==0,n.subtreeFlags&15990||l){l=gt.transition,gt.transition=null;var o=je;je=1;var c=ge;ge|=4,si.current=null,Rm(e,n),Hu(n,e),sm(Ql),ls=!!Gl,Ql=Gl=null,e.current=n,Mm(n),df(),ge=c,je=o,gt.transition=l}else e.current=n;if($a&&($a=!1,lr=e,ws=s),l=e.pendingLanes,l===0&&(pr=null),ff(n.stateNode),rt(e,Ee()),t!==null)for(a=e.onRecoverableError,n=0;n<t.length;n++)s=t[n],a(s.value,{componentStack:s.stack,digest:s.digest});if(bs)throw bs=!1,e=mo,mo=null,e;return ws&1&&e.tag!==0&&sn(),l=e.pendingLanes,l&1?e===ho?Gn++:(Gn=0,ho=e):Gn=0,wr(),null}function sn(){if(lr!==null){var e=Pd(ws),t=gt.transition,n=je;try{if(gt.transition=null,je=16>e?16:e,lr===null)var a=!1;else{if(e=lr,lr=null,ws=0,ge&6)throw Error(B(331));var s=ge;for(ge|=4,se=e.current;se!==null;){var l=se,o=l.child;if(se.flags&16){var c=l.deletions;if(c!==null){for(var d=0;d<c.length;d++){var p=c[d];for(se=p;se!==null;){var v=se;switch(v.tag){case 0:case 11:case 15:Wn(8,v,l)}var g=v.child;if(g!==null)g.return=v,se=g;else for(;se!==null;){v=se;var x=v.sibling,k=v.return;if(Vu(v),v===p){se=null;break}if(x!==null){x.return=k,se=x;break}se=k}}}var w=l.alternate;if(w!==null){var z=w.child;if(z!==null){w.child=null;do{var F=z.sibling;z.sibling=null,z=F}while(z!==null)}}se=l}}if(l.subtreeFlags&2064&&o!==null)o.return=l,se=o;else e:for(;se!==null;){if(l=se,l.flags&2048)switch(l.tag){case 0:case 11:case 15:Wn(9,l,l.return)}var f=l.sibling;if(f!==null){f.return=l.return,se=f;break e}se=l.return}}var u=e.current;for(se=u;se!==null;){o=se;var h=o.child;if(o.subtreeFlags&2064&&h!==null)h.return=o,se=h;else e:for(o=u;se!==null;){if(c=se,c.flags&2048)try{switch(c.tag){case 0:case 11:case 15:Ls(9,c)}}catch(j){ze(c,c.return,j)}if(c===o){se=null;break e}var y=c.sibling;if(y!==null){y.return=c.return,se=y;break e}se=c.return}}if(ge=s,wr(),Dt&&typeof Dt.onPostCommitFiberRoot=="function")try{Dt.onPostCommitFiberRoot(zs,e)}catch{}a=!0}return a}finally{je=n,gt.transition=t}}return!1}function _c(e,t,n){t=pn(n,t),t=Pu(e,t,1),e=ur(e,t,1),t=qe(),e!==null&&(ha(e,1,t),rt(e,t))}function ze(e,t,n){if(e.tag===3)_c(e,e,n);else for(;t!==null;){if(t.tag===3){_c(t,e,n);break}else if(t.tag===1){var a=t.stateNode;if(typeof t.type.getDerivedStateFromError=="function"||typeof a.componentDidCatch=="function"&&(pr===null||!pr.has(a))){e=pn(n,e),e=Iu(t,e,1),t=ur(t,e,1),e=qe(),t!==null&&(ha(t,1,e),rt(t,e));break}}t=t.return}}function $m(e,t,n){var a=e.pingCache;a!==null&&a.delete(t),t=qe(),e.pingedLanes|=e.suspendedLanes&n,Fe===e&&(De&n)===n&&(Re===4||Re===3&&(De&130023424)===De&&500>Ee()-oi?Tr(e,0):li|=n),rt(e,t)}function Ju(e,t){t===0&&(e.mode&1?(t=Ta,Ta<<=1,!(Ta&130023424)&&(Ta=4194304)):t=1);var n=qe();e=Yt(e,t),e!==null&&(ha(e,t,n),rt(e,n))}function Um(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),Ju(e,n)}function Vm(e,t){var n=0;switch(e.tag){case 13:var a=e.stateNode,s=e.memoizedState;s!==null&&(n=s.retryLane);break;case 19:a=e.stateNode;break;default:throw Error(B(314))}a!==null&&a.delete(t),Ju(e,n)}var Zu;Zu=function(e,t,n){if(e!==null)if(e.memoizedProps!==t.pendingProps||et.current)Ze=!0;else{if(!(e.lanes&n)&&!(t.flags&128))return Ze=!1,Em(e,t,n);Ze=!!(e.flags&131072)}else Ze=!1,Se&&t.flags&1048576&&nu(t,fs,t.index);switch(t.lanes=0,t.tag){case 2:var a=t.type;Ka(e,t),e=t.pendingProps;var s=on(t,Be.current);an(t,n),s=ei(null,t,a,e,s,n);var l=ti();return t.flags|=1,typeof s=="object"&&s!==null&&typeof s.render=="function"&&s.$$typeof===void 0?(t.tag=1,t.memoizedState=null,t.updateQueue=null,tt(a)?(l=!0,us(t)):l=!1,t.memoizedState=s.state!==null&&s.state!==void 0?s.state:null,Yo(t),s.updater=Fs,t.stateNode=s,s._reactInternals=t,ro(t,a,e,n),t=so(null,t,a,!0,l,n)):(t.tag=0,Se&&l&&Vo(t),Ge(null,t,s,n),t=t.child),t;case 16:a=t.elementType;e:{switch(Ka(e,t),e=t.pendingProps,s=a._init,a=s(a._payload),t.type=a,s=t.tag=Wm(a),e=St(a,e),s){case 0:t=ao(null,t,a,e,n);break e;case 1:t=gc(null,t,a,e,n);break e;case 11:t=hc(null,t,a,e,n);break e;case 14:t=xc(null,t,a,St(a.type,e),n);break e}throw Error(B(306,a,""))}return t;case 0:return a=t.type,s=t.pendingProps,s=t.elementType===a?s:St(a,s),ao(e,t,a,s,n);case 1:return a=t.type,s=t.pendingProps,s=t.elementType===a?s:St(a,s),gc(e,t,a,s,n);case 3:e:{if(Lu(t),e===null)throw Error(B(387));a=t.pendingProps,l=t.memoizedState,s=l.element,cu(e,t),xs(t,a,null,n);var o=t.memoizedState;if(a=o.element,l.isDehydrated)if(l={element:a,isDehydrated:!1,cache:o.cache,pendingSuspenseBoundaries:o.pendingSuspenseBoundaries,transitions:o.transitions},t.updateQueue.baseState=l,t.memoizedState=l,t.flags&256){s=pn(Error(B(423)),t),t=vc(e,t,a,n,s);break e}else if(a!==s){s=pn(Error(B(424)),t),t=vc(e,t,a,n,s);break e}else for(it=dr(t.stateNode.containerInfo.firstChild),ct=t,Se=!0,Ct=null,n=ou(t,null,a,n),t.child=n;n;)n.flags=n.flags&-3|4096,n=n.sibling;else{if(cn(),a===s){t=Xt(e,t,n);break e}Ge(e,t,a,n)}t=t.child}return t;case 5:return du(t),e===null&&Zl(t),a=t.type,s=t.pendingProps,l=e!==null?e.memoizedProps:null,o=s.children,ql(a,s)?o=null:l!==null&&ql(a,l)&&(t.flags|=32),Fu(e,t),Ge(e,t,o,n),t.child;case 6:return e===null&&Zl(t),null;case 13:return Du(e,t,n);case 4:return Xo(t,t.stateNode.containerInfo),a=t.pendingProps,e===null?t.child=dn(t,null,a,n):Ge(e,t,a,n),t.child;case 11:return a=t.type,s=t.pendingProps,s=t.elementType===a?s:St(a,s),hc(e,t,a,s,n);case 7:return Ge(e,t,t.pendingProps,n),t.child;case 8:return Ge(e,t,t.pendingProps.children,n),t.child;case 12:return Ge(e,t,t.pendingProps.children,n),t.child;case 10:e:{if(a=t.type._context,s=t.pendingProps,l=t.memoizedProps,o=s.value,be(ms,a._currentValue),a._currentValue=o,l!==null)if(Et(l.value,o)){if(l.children===s.children&&!et.current){t=Xt(e,t,n);break e}}else for(l=t.child,l!==null&&(l.return=t);l!==null;){var c=l.dependencies;if(c!==null){o=l.child;for(var d=c.firstContext;d!==null;){if(d.context===a){if(l.tag===1){d=Ht(-1,n&-n),d.tag=2;var p=l.updateQueue;if(p!==null){p=p.shared;var v=p.pending;v===null?d.next=d:(d.next=v.next,v.next=d),p.pending=d}}l.lanes|=n,d=l.alternate,d!==null&&(d.lanes|=n),eo(l.return,n,t),c.lanes|=n;break}d=d.next}}else if(l.tag===10)o=l.type===t.type?null:l.child;else if(l.tag===18){if(o=l.return,o===null)throw Error(B(341));o.lanes|=n,c=o.alternate,c!==null&&(c.lanes|=n),eo(o,n,t),o=l.sibling}else o=l.child;if(o!==null)o.return=l;else for(o=l;o!==null;){if(o===t){o=null;break}if(l=o.sibling,l!==null){l.return=o.return,o=l;break}o=o.return}l=o}Ge(e,t,s.children,n),t=t.child}return t;case 9:return s=t.type,a=t.pendingProps.children,an(t,n),s=yt(s),a=a(s),t.flags|=1,Ge(e,t,a,n),t.child;case 14:return a=t.type,s=St(a,t.pendingProps),s=St(a.type,s),xc(e,t,a,s,n);case 15:return Ru(e,t,t.type,t.pendingProps,n);case 17:return a=t.type,s=t.pendingProps,s=t.elementType===a?s:St(a,s),Ka(e,t),t.tag=1,tt(a)?(e=!0,us(t)):e=!1,an(t,n),Tu(t,a,s),ro(t,a,s,n),so(null,t,a,!0,e,n);case 19:return Ou(e,t,n);case 22:return Mu(e,t,n)}throw Error(B(156,t.tag))};function ep(e,t){return _d(e,t)}function Bm(e,t,n,a){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=a,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function xt(e,t,n,a){return new Bm(e,t,n,a)}function ui(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Wm(e){if(typeof e=="function")return ui(e)?1:0;if(e!=null){if(e=e.$$typeof,e===To)return 11;if(e===Po)return 14}return 2}function mr(e,t){var n=e.alternate;return n===null?(n=xt(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&14680064,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n}function es(e,t,n,a,s,l){var o=2;if(a=e,typeof e=="function")ui(e)&&(o=1);else if(typeof e=="string")o=5;else e:switch(e){case Br:return Pr(n.children,s,l,t);case Eo:o=8,s|=8;break;case Cl:return e=xt(12,n,t,s|2),e.elementType=Cl,e.lanes=l,e;case _l:return e=xt(13,n,t,s),e.elementType=_l,e.lanes=l,e;case zl:return e=xt(19,n,t,s),e.elementType=zl,e.lanes=l,e;case dd:return Os(n,s,l,t);default:if(typeof e=="object"&&e!==null)switch(e.$$typeof){case id:o=10;break e;case cd:o=9;break e;case To:o=11;break e;case Po:o=14;break e;case tr:o=16,a=null;break e}throw Error(B(130,e==null?e:typeof e,""))}return t=xt(o,n,t,s),t.elementType=e,t.type=a,t.lanes=l,t}function Pr(e,t,n,a){return e=xt(7,e,a,t),e.lanes=n,e}function Os(e,t,n,a){return e=xt(22,e,a,t),e.elementType=dd,e.lanes=n,e.stateNode={isHidden:!1},e}function gl(e,t,n){return e=xt(6,e,null,t),e.lanes=n,e}function vl(e,t,n){return t=xt(4,e.children!==null?e.children:[],e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}function Hm(e,t,n,a,s){this.tag=t,this.containerInfo=e,this.finishedWork=this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.pendingContext=this.context=null,this.callbackPriority=0,this.eventTimes=Js(0),this.expirationTimes=Js(-1),this.entangledLanes=this.finishedLanes=this.mutableReadLanes=this.expiredLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=Js(0),this.identifierPrefix=a,this.onRecoverableError=s,this.mutableSourceEagerHydrationData=null}function pi(e,t,n,a,s,l,o,c,d){return e=new Hm(e,t,n,c,d),t===1?(t=1,l===!0&&(t|=8)):t=0,l=xt(3,null,null,t),e.current=l,l.stateNode=e,l.memoizedState={element:a,isDehydrated:n,cache:null,transitions:null,pendingSuspenseBoundaries:null},Yo(l),e}function Gm(e,t,n){var a=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:Vr,key:a==null?null:""+a,children:e,containerInfo:t,implementation:n}}function tp(e){if(!e)return xr;e=e._reactInternals;e:{if(Or(e)!==e||e.tag!==1)throw Error(B(170));var t=e;do{switch(t.tag){case 3:t=t.stateNode.context;break e;case 1:if(tt(t.type)){t=t.stateNode.__reactInternalMemoizedMergedChildContext;break e}}t=t.return}while(t!==null);throw Error(B(171))}if(e.tag===1){var n=e.type;if(tt(n))return tu(e,n,t)}return t}function rp(e,t,n,a,s,l,o,c,d){return e=pi(n,a,!0,e,s,l,o,c,d),e.context=tp(null),n=e.current,a=qe(),s=fr(n),l=Ht(a,s),l.callback=t??null,ur(n,l,s),e.current.lanes=s,ha(e,s,a),rt(e,a),e}function As(e,t,n,a){var s=t.current,l=qe(),o=fr(s);return n=tp(n),t.context===null?t.context=n:t.pendingContext=n,t=Ht(l,o),t.payload={element:e},a=a===void 0?null:a,a!==null&&(t.callback=a),e=ur(s,t,o),e!==null&&(zt(e,s,o,l),qa(e,s,o)),o}function Ss(e){if(e=e.current,!e.child)return null;switch(e.child.tag){case 5:return e.child.stateNode;default:return e.child.stateNode}}function zc(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function fi(e,t){zc(e,t),(e=e.alternate)&&zc(e,t)}function Qm(){return null}var np=typeof reportError=="function"?reportError:function(e){console.error(e)};function mi(e){this._internalRoot=e}$s.prototype.render=mi.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(B(409));As(e,t,null,null)};$s.prototype.unmount=mi.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;Lr(function(){As(null,e,null,null)}),t[qt]=null}};function $s(e){this._internalRoot=e}$s.prototype.unstable_scheduleHydration=function(e){if(e){var t=Md();e={blockedOn:null,target:e,priority:t};for(var n=0;n<nr.length&&t!==0&&t<nr[n].priority;n++);nr.splice(n,0,e),n===0&&Ld(e)}};function hi(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function Us(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11&&(e.nodeType!==8||e.nodeValue!==" react-mount-point-unstable "))}function Ec(){}function qm(e,t,n,a,s){if(s){if(typeof a=="function"){var l=a;a=function(){var p=Ss(o);l.call(p)}}var o=rp(t,a,e,0,null,!1,!1,"",Ec);return e._reactRootContainer=o,e[qt]=o.current,ta(e.nodeType===8?e.parentNode:e),Lr(),o}for(;s=e.lastChild;)e.removeChild(s);if(typeof a=="function"){var c=a;a=function(){var p=Ss(d);c.call(p)}}var d=pi(e,0,!1,null,null,!1,!1,"",Ec);return e._reactRootContainer=d,e[qt]=d.current,ta(e.nodeType===8?e.parentNode:e),Lr(function(){As(t,d,n,a)}),d}function Vs(e,t,n,a,s){var l=n._reactRootContainer;if(l){var o=l;if(typeof s=="function"){var c=s;s=function(){var d=Ss(o);c.call(d)}}As(t,o,e,s)}else o=qm(n,t,e,s,a);return Ss(o)}Id=function(e){switch(e.tag){case 3:var t=e.stateNode;if(t.current.memoizedState.isDehydrated){var n=Ln(t.pendingLanes);n!==0&&(Mo(t,n|1),rt(t,Ee()),!(ge&6)&&(fn=Ee()+500,wr()))}break;case 13:Lr(function(){var a=Yt(e,1);if(a!==null){var s=qe();zt(a,e,1,s)}}),fi(e,1)}};Fo=function(e){if(e.tag===13){var t=Yt(e,134217728);if(t!==null){var n=qe();zt(t,e,134217728,n)}fi(e,134217728)}};Rd=function(e){if(e.tag===13){var t=fr(e),n=Yt(e,t);if(n!==null){var a=qe();zt(n,e,t,a)}fi(e,t)}};Md=function(){return je};Fd=function(e,t){var n=je;try{return je=e,t()}finally{je=n}};Ol=function(e,t,n){switch(t){case"input":if(Pl(e,n),t=n.name,n.type==="radio"&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll("input[name="+JSON.stringify(""+t)+'][type="radio"]'),t=0;t<n.length;t++){var a=n[t];if(a!==e&&a.form===e.form){var s=Is(a);if(!s)throw Error(B(90));pd(a),Pl(a,s)}}}break;case"textarea":md(e,n);break;case"select":t=n.value,t!=null&&en(e,!!n.multiple,t,!1)}};bd=ii;wd=Lr;var Ym={usingClientEntryPoint:!1,Events:[ga,Qr,Is,yd,jd,ii]},Rn={findFiberByHostInstance:_r,bundleType:0,version:"18.3.1",rendererPackageName:"react-dom"},Xm={bundleType:Rn.bundleType,version:Rn.version,rendererPackageName:Rn.rendererPackageName,rendererConfig:Rn.rendererConfig,overrideHookState:null,overrideHookStateDeletePath:null,overrideHookStateRenamePath:null,overrideProps:null,overridePropsDeletePath:null,overridePropsRenamePath:null,setErrorHandler:null,setSuspenseHandler:null,scheduleUpdate:null,currentDispatcherRef:Jt.ReactCurrentDispatcher,findHostInstanceByFiber:function(e){return e=Nd(e),e===null?null:e.stateNode},findFiberByHostInstance:Rn.findFiberByHostInstance||Qm,findHostInstancesForRefresh:null,scheduleRefresh:null,scheduleRoot:null,setRefreshHandler:null,getCurrentFiber:null,reconcilerVersion:"18.3.1-next-f1338f8080-20240426"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"){var Ua=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Ua.isDisabled&&Ua.supportsFiber)try{zs=Ua.inject(Xm),Dt=Ua}catch{}}ut.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=Ym;ut.createPortal=function(e,t){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!hi(t))throw Error(B(200));return Gm(e,t,null,n)};ut.createRoot=function(e,t){if(!hi(e))throw Error(B(299));var n=!1,a="",s=np;return t!=null&&(t.unstable_strictMode===!0&&(n=!0),t.identifierPrefix!==void 0&&(a=t.identifierPrefix),t.onRecoverableError!==void 0&&(s=t.onRecoverableError)),t=pi(e,1,!1,null,null,n,!1,a,s),e[qt]=t.current,ta(e.nodeType===8?e.parentNode:e),new mi(t)};ut.findDOMNode=function(e){if(e==null)return null;if(e.nodeType===1)return e;var t=e._reactInternals;if(t===void 0)throw typeof e.render=="function"?Error(B(188)):(e=Object.keys(e).join(","),Error(B(268,e)));return e=Nd(t),e=e===null?null:e.stateNode,e};ut.flushSync=function(e){return Lr(e)};ut.hydrate=function(e,t,n){if(!Us(t))throw Error(B(200));return Vs(null,e,t,!0,n)};ut.hydrateRoot=function(e,t,n){if(!hi(e))throw Error(B(405));var a=n!=null&&n.hydratedSources||null,s=!1,l="",o=np;if(n!=null&&(n.unstable_strictMode===!0&&(s=!0),n.identifierPrefix!==void 0&&(l=n.identifierPrefix),n.onRecoverableError!==void 0&&(o=n.onRecoverableError)),t=rp(t,null,e,1,n??null,s,!1,l,o),e[qt]=t.current,ta(e),a)for(e=0;e<a.length;e++)n=a[e],s=n._getVersion,s=s(n._source),t.mutableSourceEagerHydrationData==null?t.mutableSourceEagerHydrationData=[n,s]:t.mutableSourceEagerHydrationData.push(n,s);return new $s(t)};ut.render=function(e,t,n){if(!Us(t))throw Error(B(200));return Vs(null,e,t,!1,n)};ut.unmountComponentAtNode=function(e){if(!Us(e))throw Error(B(40));return e._reactRootContainer?(Lr(function(){Vs(null,null,e,!1,function(){e._reactRootContainer=null,e[qt]=null})}),!0):!1};ut.unstable_batchedUpdates=ii;ut.unstable_renderSubtreeIntoContainer=function(e,t,n,a){if(!Us(n))throw Error(B(200));if(e==null||e._reactInternals===void 0)throw Error(B(38));return Vs(e,t,n,!1,a)};ut.version="18.3.1-next-f1338f8080-20240426";function ap(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(ap)}catch(e){console.error(e)}}ap(),ad.exports=ut;var Km=ad.exports,Tc=Km;Sl.createRoot=Tc.createRoot,Sl.hydrateRoot=Tc.hydrateRoot;/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Jm=e=>e.replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase(),sp=(...e)=>e.filter((t,n,a)=>!!t&&t.trim()!==""&&a.indexOf(t)===n).join(" ").trim();/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */var Zm={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const eh=i.forwardRef(({color:e="currentColor",size:t=24,strokeWidth:n=2,absoluteStrokeWidth:a,className:s="",children:l,iconNode:o,...c},d)=>i.createElement("svg",{ref:d,...Zm,width:t,height:t,stroke:e,strokeWidth:a?Number(n)*24/Number(t):n,className:sp("lucide",s),...c},[...o.map(([p,v])=>i.createElement(p,v)),...Array.isArray(l)?l:[l]]));/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const re=(e,t)=>{const n=i.forwardRef(({className:a,...s},l)=>i.createElement(eh,{ref:l,iconNode:t,className:sp(`lucide-${Jm(e)}`,a),...s}));return n.displayName=`${e}`,n};/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const th=[["path",{d:"M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2",key:"169zse"}]],rh=re("Activity",th);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const nh=[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"m12 5 7 7-7 7",key:"xquz4c"}]],ah=re("ArrowRight",nh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const sh=[["path",{d:"m21 16-4 4-4-4",key:"f6ql7i"}],["path",{d:"M17 20V4",key:"1ejh1v"}],["path",{d:"m3 8 4-4 4 4",key:"11wl7u"}],["path",{d:"M7 4v16",key:"1glfcx"}]],lh=re("ArrowUpDown",sh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const oh=[["path",{d:"M20 6 9 17l-5-5",key:"1gmf2c"}]],Ns=re("Check",oh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ih=[["path",{d:"m6 9 6 6 6-6",key:"qrunsl"}]],Tt=re("ChevronDown",ih);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ch=[["path",{d:"m15 18-6-6 6-6",key:"1wnfg3"}]],lp=re("ChevronLeft",ch);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const dh=[["path",{d:"m9 18 6-6-6-6",key:"mthhwq"}]],op=re("ChevronRight",dh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const uh=[["path",{d:"m18 15-6-6-6 6",key:"153udz"}]],ph=re("ChevronUp",uh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const fh=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["line",{x1:"12",x2:"12",y1:"8",y2:"12",key:"1pkeuh"}],["line",{x1:"12",x2:"12.01",y1:"16",y2:"16",key:"4dfq90"}]],mh=re("CircleAlert",fh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const hh=[["path",{d:"M21.801 10A10 10 0 1 1 17 3.335",key:"yps3ct"}],["path",{d:"m9 11 3 3L22 4",key:"1pflzl"}]],xh=re("CircleCheckBig",hh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const gh=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"m9 12 2 2 4-4",key:"dzmm74"}]],vh=re("CircleCheck",gh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const yh=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3",key:"1u773s"}],["path",{d:"M12 17h.01",key:"p32p05"}]],ip=re("CircleHelp",yh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const jh=[["path",{d:"M20.2 6 3 11l-.9-2.4c-.3-1.1.3-2.2 1.3-2.5l13.5-4c1.1-.3 2.2.3 2.5 1.3Z",key:"1tn4o7"}],["path",{d:"m6.2 5.3 3.1 3.9",key:"iuk76l"}],["path",{d:"m12.4 3.4 3.1 4",key:"6hsd6n"}],["path",{d:"M3 11h18v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z",key:"ltgou9"}]],bh=re("Clapperboard",jh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const wh=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["polyline",{points:"12 6 12 12 16 14",key:"68esgv"}]],Bs=re("Clock",wh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const kh=[["rect",{width:"14",height:"14",x:"8",y:"8",rx:"2",ry:"2",key:"17jyea"}],["path",{d:"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2",key:"zix9uf"}]],Wt=re("Copy",kh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Sh=[["rect",{width:"16",height:"16",x:"4",y:"4",rx:"2",key:"14l7u7"}],["rect",{width:"6",height:"6",x:"9",y:"9",rx:"1",key:"5aljv4"}],["path",{d:"M15 2v2",key:"13l42r"}],["path",{d:"M15 20v2",key:"15mkzm"}],["path",{d:"M2 15h2",key:"1gxd5l"}],["path",{d:"M2 9h2",key:"1bbxkp"}],["path",{d:"M20 15h2",key:"19e6y8"}],["path",{d:"M20 9h2",key:"19tzq7"}],["path",{d:"M9 2v2",key:"165o2o"}],["path",{d:"M9 20v2",key:"i2bqo8"}]],Nh=re("Cpu",Sh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ch=[["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",key:"ih7n3h"}],["polyline",{points:"7 10 12 15 17 10",key:"2ggqvy"}],["line",{x1:"12",x2:"12",y1:"15",y2:"3",key:"1vk2je"}]],vt=re("Download",Ch);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const _h=[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]],Pc=re("ExternalLink",_h);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const zh=[["path",{d:"M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49",key:"ct8e1f"}],["path",{d:"M14.084 14.158a3 3 0 0 1-4.242-4.242",key:"151rxh"}],["path",{d:"M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143",key:"13bj9a"}],["path",{d:"m2 2 20 20",key:"1ooewy"}]],Eh=re("EyeOff",zh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Th=[["path",{d:"M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0",key:"1nclc0"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]],Ph=re("Eye",Th);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ih=[["path",{d:"M17.5 22h.5a2 2 0 0 0 2-2V7l-5-5H6a2 2 0 0 0-2 2v3",key:"rslqgf"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"M2 19a2 2 0 1 1 4 0v1a2 2 0 1 1-4 0v-4a6 6 0 0 1 12 0v4a2 2 0 1 1-4 0v-1a2 2 0 1 1 4 0",key:"9f7x3i"}]],vo=re("FileAudio",Ih);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Rh=[["path",{d:"M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",key:"1rqfz7"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"M10 12a1 1 0 0 0-1 1v1a1 1 0 0 1-1 1 1 1 0 0 1 1 1v1a1 1 0 0 0 1 1",key:"1oajmo"}],["path",{d:"M14 18a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1 1 1 0 0 1-1-1v-1a1 1 0 0 0-1-1",key:"mpwhp6"}]],Ic=re("FileJson",Rh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Mh=[["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"M4.268 21a2 2 0 0 0 1.727 1H18a2 2 0 0 0 2-2V7l-5-5H6a2 2 0 0 0-2 2v3",key:"ms7g94"}],["path",{d:"m9 18-1.5-1.5",key:"1j6qii"}],["circle",{cx:"5",cy:"14",r:"3",key:"ufru5t"}]],Fh=re("FileSearch",Mh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Lh=[["path",{d:"M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",key:"1rqfz7"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"M10 9H8",key:"b1mrlr"}],["path",{d:"M16 13H8",key:"t4e002"}],["path",{d:"M16 17H8",key:"z1uh3a"}]],Rc=re("FileText",Lh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Dh=[["path",{d:"M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",key:"1rqfz7"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"m10 11 5 3-5 3v-6Z",key:"7ntvm4"}]],Oh=re("FileVideo",Dh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ah=[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",key:"afitv7"}],["path",{d:"M7 3v18",key:"bbkbws"}],["path",{d:"M3 7.5h4",key:"zfgn84"}],["path",{d:"M3 12h18",key:"1i2n21"}],["path",{d:"M3 16.5h4",key:"1230mu"}],["path",{d:"M17 3v18",key:"in4fa5"}],["path",{d:"M17 7.5h4",key:"myr1c1"}],["path",{d:"M17 16.5h4",key:"go4c1d"}]],da=re("Film",Ah);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const $h=[["polygon",{points:"22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3",key:"1yg77f"}]],Uh=re("Filter",$h);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Vh=[["path",{d:"m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2",key:"usdka0"}]],cp=re("FolderOpen",Vh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Bh=[["line",{x1:"22",x2:"2",y1:"6",y2:"6",key:"15w7dq"}],["line",{x1:"22",x2:"2",y1:"18",y2:"18",key:"1ip48p"}],["line",{x1:"6",x2:"6",y1:"2",y2:"22",key:"a2lnyx"}],["line",{x1:"18",x2:"18",y1:"2",y2:"22",key:"8vb6jd"}]],Wh=re("Frame",Bh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Hh=[["path",{d:"M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z",key:"c3ymky"}]],yl=re("Heart",Hh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Gh=[["path",{d:"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8",key:"1357e3"}],["path",{d:"M3 3v5h5",key:"1xhq8a"}],["path",{d:"M12 7v5l4 2",key:"1fdv2h"}]],Qh=re("History",Gh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const qh=[["path",{d:"M16 5h6",key:"1vod17"}],["path",{d:"M19 2v6",key:"4bpg5p"}],["path",{d:"M21 11.5V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7.5",key:"1ue2ih"}],["path",{d:"m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21",key:"1xmnt7"}],["circle",{cx:"9",cy:"9",r:"2",key:"af1f0g"}]],Yh=re("ImagePlus",qh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Xh=[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",ry:"2",key:"1m3agn"}],["circle",{cx:"9",cy:"9",r:"2",key:"af1f0g"}],["path",{d:"m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21",key:"1xmnt7"}]],gr=re("Image",Xh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Kh=[["path",{d:"M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z",key:"zw3jo"}],["path",{d:"M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12",key:"1wduqc"}],["path",{d:"M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17",key:"kqbvx6"}]],xi=re("Layers",Kh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Jh=[["path",{d:"M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71",key:"1cjeqo"}],["path",{d:"M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",key:"19qd67"}]],dp=re("Link",Jh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Zh=[["path",{d:"M21 12a9 9 0 1 1-6.219-8.56",key:"13zald"}]],Oe=re("LoaderCircle",Zh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ex=[["polyline",{points:"15 3 21 3 21 9",key:"mznyad"}],["polyline",{points:"9 21 3 21 3 15",key:"1avn1i"}],["line",{x1:"21",x2:"14",y1:"3",y2:"10",key:"ota7mn"}],["line",{x1:"3",x2:"10",y1:"21",y2:"14",key:"1atl0r"}]],up=re("Maximize2",ex);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const tx=[["path",{d:"M7.9 20A9 9 0 1 0 4 16.1L2 22Z",key:"vv11sd"}]],rx=re("MessageCircle",tx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const nx=[["path",{d:"M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",key:"1lielz"}]],jl=re("MessageSquare",nx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ax=[["path",{d:"M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z",key:"131961"}],["path",{d:"M19 10v2a7 7 0 0 1-14 0v-2",key:"1vc78b"}],["line",{x1:"12",x2:"12",y1:"19",y2:"22",key:"x3vr5v"}]],gi=re("Mic",ax);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const sx=[["polyline",{points:"4 14 10 14 10 20",key:"11kfnr"}],["polyline",{points:"20 10 14 10 14 4",key:"rlmsce"}],["line",{x1:"14",x2:"21",y1:"10",y2:"3",key:"o5lafz"}],["line",{x1:"3",x2:"10",y1:"21",y2:"14",key:"1atl0r"}]],lx=re("Minimize2",sx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ox=[["path",{d:"M12 2v20",key:"t6zp3m"}],["path",{d:"m15 19-3 3-3-3",key:"11eu04"}],["path",{d:"m19 9 3 3-3 3",key:"1mg7y2"}],["path",{d:"M2 12h20",key:"9i4pu4"}],["path",{d:"m5 9-3 3 3 3",key:"j64kie"}],["path",{d:"m9 5 3-3 3 3",key:"l8vdw6"}]],ix=re("Move",ox);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const cx=[["path",{d:"M9 18V5l12-2v13",key:"1jmyc2"}],["circle",{cx:"6",cy:"18",r:"3",key:"fqmcym"}],["circle",{cx:"18",cy:"16",r:"3",key:"1hluhg"}]],dx=re("Music",cx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ux=[["rect",{x:"14",y:"4",width:"4",height:"16",rx:"1",key:"zuxfzm"}],["rect",{x:"6",y:"4",width:"4",height:"16",rx:"1",key:"1okwgv"}]],yo=re("Pause",ux);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const px=[["polygon",{points:"6 3 20 12 6 21 6 3",key:"1oa8hb"}]],ua=re("Play",px);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const fx=[["path",{d:"M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8",key:"v9h5vc"}],["path",{d:"M21 3v5h-5",key:"1q7to0"}],["path",{d:"M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16",key:"3uifl3"}],["path",{d:"M8 16H3v5",key:"1cv678"}]],mn=re("RefreshCw",fx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const mx=[["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}],["path",{d:"m21 21-4.3-4.3",key:"1qie3q"}]],hx=re("Search",mx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const xx=[["path",{d:"M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z",key:"1ffxy3"}],["path",{d:"m21.854 2.147-10.94 10.939",key:"12cjpa"}]],pp=re("Send",xx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const gx=[["path",{d:"M20 7h-9",key:"3s1dr2"}],["path",{d:"M14 17H5",key:"gfn3mx"}],["circle",{cx:"17",cy:"17",r:"3",key:"18b49y"}],["circle",{cx:"7",cy:"7",r:"3",key:"dfmy0x"}]],fp=re("Settings2",gx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const vx=[["path",{d:"M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z",key:"1qme2f"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]],vr=re("Settings",vx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const yx=[["line",{x1:"4",x2:"4",y1:"21",y2:"14",key:"1p332r"}],["line",{x1:"4",x2:"4",y1:"10",y2:"3",key:"gb41h5"}],["line",{x1:"12",x2:"12",y1:"21",y2:"12",key:"hf2csr"}],["line",{x1:"12",x2:"12",y1:"8",y2:"3",key:"1kfi7u"}],["line",{x1:"20",x2:"20",y1:"21",y2:"16",key:"1lhrwl"}],["line",{x1:"20",x2:"20",y1:"12",y2:"3",key:"16vvfq"}],["line",{x1:"2",x2:"6",y1:"14",y2:"14",key:"1uebub"}],["line",{x1:"10",x2:"14",y1:"8",y2:"8",key:"1yglbp"}],["line",{x1:"18",x2:"22",y1:"16",y2:"16",key:"1jxqpz"}]],pa=re("SlidersVertical",yx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const jx=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M8 14s1.5 2 4 2 4-2 4-2",key:"1y1vjs"}],["line",{x1:"9",x2:"9.01",y1:"9",y2:"9",key:"yxxnd0"}],["line",{x1:"15",x2:"15.01",y1:"9",y2:"9",key:"1p4y9e"}]],bx=re("Smile",jx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const wx=[["path",{d:"M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z",key:"4pj2yx"}],["path",{d:"M20 3v4",key:"1olli1"}],["path",{d:"M22 5h-4",key:"1gvqau"}],["path",{d:"M4 17v2",key:"vumght"}],["path",{d:"M5 18H3",key:"zchphs"}]],Gt=re("Sparkles",wx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const kx=[["polyline",{points:"4 17 10 11 4 5",key:"akl6gq"}],["line",{x1:"12",x2:"20",y1:"19",y2:"19",key:"q2wloq"}]],Mc=re("Terminal",kx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Sx=[["path",{d:"M3 6h18",key:"d0wm0j"}],["path",{d:"M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6",key:"4alrt4"}],["path",{d:"M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2",key:"v07s0e"}],["line",{x1:"10",x2:"10",y1:"11",y2:"17",key:"1uufr5"}],["line",{x1:"14",x2:"14",y1:"11",y2:"17",key:"xtxkd"}]],Cs=re("Trash2",Sx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Nx=[["polyline",{points:"4 7 4 4 20 4 20 7",key:"1nosan"}],["line",{x1:"9",x2:"15",y1:"20",y2:"20",key:"swin9y"}],["line",{x1:"12",x2:"12",y1:"4",y2:"20",key:"1tx1rr"}]],mp=re("Type",Nx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Cx=[["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",key:"ih7n3h"}],["polyline",{points:"17 8 12 3 7 8",key:"t8dd8p"}],["line",{x1:"12",x2:"12",y1:"3",y2:"15",key:"widbto"}]],Ye=re("Upload",Cx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const _x=[["path",{d:"M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2",key:"975kel"}],["circle",{cx:"12",cy:"7",r:"4",key:"17ys0d"}]],jo=re("User",_x);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const zx=[["path",{d:"m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5",key:"ftymec"}],["rect",{x:"2",y:"6",width:"14",height:"12",rx:"2",key:"158x01"}]],yr=re("Video",zx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ex=[["path",{d:"M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z",key:"uqj9uw"}],["path",{d:"M16 9a5 5 0 0 1 0 6",key:"1q6k2b"}],["path",{d:"M19.364 18.364a9 9 0 0 0 0-12.728",key:"ijwkga"}]],hn=re("Volume2",Ex);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Tx=[["path",{d:"m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72",key:"ul74o6"}],["path",{d:"m14 7 3 3",key:"1r5n42"}],["path",{d:"M5 6v4",key:"ilb8ba"}],["path",{d:"M19 14v4",key:"blhpug"}],["path",{d:"M10 2v2",key:"7u0qdc"}],["path",{d:"M7 8H3",key:"zfb6yr"}],["path",{d:"M21 16h-4",key:"1cnmox"}],["path",{d:"M11 3H9",key:"1obp7u"}]],Kt=re("WandSparkles",Tx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Px=[["path",{d:"M12 20h.01",key:"zekei9"}],["path",{d:"M8.5 16.429a5 5 0 0 1 7 0",key:"1bycff"}],["path",{d:"M5 12.859a10 10 0 0 1 5.17-2.69",key:"1dl1wf"}],["path",{d:"M19 12.859a10 10 0 0 0-2.007-1.523",key:"4k23kn"}],["path",{d:"M2 8.82a15 15 0 0 1 4.177-2.643",key:"1grhjp"}],["path",{d:"M22 8.82a15 15 0 0 0-11.288-3.764",key:"z3jwby"}],["path",{d:"m2 2 20 20",key:"1ooewy"}]],Ix=re("WifiOff",Px);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Rx=[["path",{d:"M12 20h.01",key:"zekei9"}],["path",{d:"M2 8.82a15 15 0 0 1 20 0",key:"dnpr2z"}],["path",{d:"M5 12.859a10 10 0 0 1 14 0",key:"1x1e6c"}],["path",{d:"M8.5 16.429a5 5 0 0 1 7 0",key:"1bycff"}]],Mx=re("Wifi",Rx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Fx=[["rect",{width:"8",height:"8",x:"3",y:"3",rx:"2",key:"by2w9f"}],["path",{d:"M7 11v4a2 2 0 0 0 2 2h4",key:"xkn7yn"}],["rect",{width:"8",height:"8",x:"13",y:"13",rx:"2",key:"1cgmvn"}]],hp=re("Workflow",Fx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Lx=[["path",{d:"M18 6 6 18",key:"1bl5f8"}],["path",{d:"m6 6 12 12",key:"d8bk6v"}]],Qe=re("X",Lx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Dx=[["path",{d:"M2.5 17a24.12 24.12 0 0 1 0-10 2 2 0 0 1 1.4-1.4 49.56 49.56 0 0 1 16.2 0A2 2 0 0 1 21.5 7a24.12 24.12 0 0 1 0 10 2 2 0 0 1-1.4 1.4 49.55 49.55 0 0 1-16.2 0A2 2 0 0 1 2.5 17",key:"1q2vi4"}],["path",{d:"m10 15 5-3-5-3z",key:"1jp15x"}]],Ox=re("Youtube",Dx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ax=[["path",{d:"M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z",key:"1xq2db"}]],xp=re("Zap",Ax);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const $x=[["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}],["line",{x1:"21",x2:"16.65",y1:"21",y2:"16.65",key:"13gj7c"}],["line",{x1:"11",x2:"11",y1:"8",y2:"14",key:"1vmskp"}],["line",{x1:"8",x2:"14",y1:"11",y2:"11",key:"durymu"}]],Fc=re("ZoomIn",$x),oe="http://192.168.1.2:7998",xe=!1,J={TEXT_TO_IMAGE:"text-to-image",TEXT_TO_VIDEO:"text-to-video",IMAGE_TO_VIDEO:"image-to-video",TEXT_TO_IMAGE_TO_VIDEO:"text-to-image-to-video",VIDEO_TO_VIDEO:"video-to-video",SPEECH_TO_VIDEO:"speech-to-video",IMAGE_TO_IMAGE:"image-to-image",REFRAME:"reframe",FACE_SWAP:"face-swap",UPSCALER:"upscaler",PROMPT_GENERATOR:"prompt-generator",IMAGE_TO_TEXT:"image-to-text",VIDEO_TO_TEXT:"video-to-text",AUDIO_GENERATION:"audio-generation",VOICE_CLONING:"voice-cloning",LIP_SYNC:"lip-sync",PIPELINE:"pipeline",LORA_TRAINING:"lora-training",MY_MEDIA_ALL:"my-media-all",MY_MEDIA_VIDEOS:"my-media-videos",MY_MEDIA_IMAGES:"my-media-images",MY_MEDIA_AUDIO:"my-media-audio",MY_MEDIA_PROMPTS:"my-media-prompts"},Ux=[{id:"video-tools",title:"Video Tools",items:[{id:J.IMAGE_TO_VIDEO,label:"Image to Video",status:"ready"},{id:J.TEXT_TO_VIDEO,label:"Text to Video",status:"ready"},{id:J.TEXT_TO_IMAGE_TO_VIDEO,label:"Text to Image to Video",status:"ready"},{id:J.VIDEO_TO_VIDEO,label:"Video to Video",status:"ready"},{id:J.SPEECH_TO_VIDEO,label:"Speech to Video",status:"new"}]},{id:"image-tools",title:"Image Tools",items:[{id:J.TEXT_TO_IMAGE,label:"Text to Image",status:"ready"},{id:J.IMAGE_TO_IMAGE,label:"Image to Image",status:"ready"},{id:J.UPSCALER,label:"Upscaler",status:"ready"},{id:J.REFRAME,label:"Reframe",status:"new"},{id:J.FACE_SWAP,label:"Face Swap",status:"new"}]},{id:"prompt-tools",title:"Prompt Tools",items:[{id:J.PROMPT_GENERATOR,label:"Prompt Generator",status:"new"},{id:J.IMAGE_TO_TEXT,label:"Image to Text",status:"new"},{id:J.VIDEO_TO_TEXT,label:"Video to Text",status:"new"}]},{id:"audio-tools",title:"Audio Tools",items:[{id:J.AUDIO_GENERATION,label:"Audio Generation",status:"new"},{id:J.VOICE_CLONING,label:"Voice Cloning",status:"new"},{id:J.LIP_SYNC,label:"Lip Sync",status:"new"}]},{id:"advanced",title:"Advanced",items:[{id:J.PIPELINE,label:"Pipeline",status:"ready"},{id:J.LORA_TRAINING,label:"LoRA Training",status:"ready"}]},{id:"my-media",title:"My Media",items:[{id:J.MY_MEDIA_ALL,label:"All",status:"ready"},{id:J.MY_MEDIA_VIDEOS,label:"Videos",status:"ready"},{id:J.MY_MEDIA_IMAGES,label:"Images",status:"ready"},{id:J.MY_MEDIA_AUDIO,label:"Audio",status:"ready"},{id:J.MY_MEDIA_PROMPTS,label:"Prompts",status:"ready"}]}],Vx={"text-to-video":yr,"image-to-video":da,"text-to-image-to-video":bh,pipeline:hp,"video-to-video":xi,"text-to-image":mp,"image-to-image":gr,reframe:up,"face-swap":jo,upscaler:Kt,"lora-training":mn,"my-media-all":cp,"my-media-videos":ua,"my-media-images":Yh};function Bx({activeToolId:e,onSelectTool:t,collapsed:n,onToggleCollapsed:a}){return r.jsxs("aside",{className:`sidebar ${n?"collapsed":""}`,children:[r.jsx("div",{className:"sidebar-header",children:r.jsx("div",{className:"sidebar-logo",children:"Oelala"})}),r.jsx("nav",{className:"sidebar-nav",children:Ux.map(s=>r.jsxs("div",{className:"sidebar-group",children:[r.jsx("div",{className:"sidebar-group-title",children:s.title}),s.items.map(l=>{const o=e===l.id,c=Vx[l.id]||Kt;return r.jsxs("button",{className:`nav-item${o?" active":""}`,onClick:()=>t(l.id),type:"button",children:[r.jsx("span",{className:"nav-icon",children:r.jsx(c,{size:18})}),r.jsx("span",{className:"nav-label",children:l.label}),l.status==="missing-backend"&&r.jsx("span",{className:"nav-badge",children:"v2"})]},l.id)})]},s.id))}),r.jsx("div",{className:"sidebar-footer",children:r.jsxs("button",{onClick:a,className:"nav-item collapse-btn",children:[r.jsx("span",{className:"nav-icon",children:n?r.jsx(op,{size:18}):r.jsx(lp,{size:18})}),r.jsx("span",{className:"nav-label",children:"Collapse"})]})})]})}async function Ws(e){try{await fetch(`${oe}/client-log`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(e)})}catch(t){console.error("Failed to send client log",t)}}function Wx(e){const[t,n]=i.useState([]),[a,s]=i.useState(!1),[l,o]=i.useState(""),c=i.useCallback(async()=>{s(!0),o("");try{const d=await fetch(`${oe}/list-videos`),p=await d.json();if(!d.ok)throw new Error((p==null?void 0:p.detail)||`History failed (${d.status})`);n(Array.isArray(p==null?void 0:p.videos)?p.videos:[])}catch(d){const p=(d==null?void 0:d.message)||"Failed to load history";o(p),await Ws({level:"error",message:"History fetch failed",timestamp:new Date().toISOString(),meta:{message:p}})}finally{s(!1)}},[]);return i.useEffect(()=>{c()},[c,e]),{videos:t,loading:a,error:l,refresh:c}}function Lc(e){const t=Math.floor(e/60),n=Math.floor(e%60);return`${t}:${n.toString().padStart(2,"0")}`}function Hx({output:e,refreshToken:t,onSelectHistoryVideo:n,onClose:a}){const[s,l]=i.useState(!1),[o,c]=i.useState(null),[d,p]=i.useState(!1),{videos:v,loading:g,error:x}=Wx(t),k=i.useMemo(()=>e?e.kind==="video"?r.jsxs("div",{className:"media-container",children:[r.jsxs("div",{className:"video-wrapper",onMouseEnter:()=>p(!0),onMouseLeave:()=>p(!1),children:[r.jsx("video",{className:"media-preview",controls:!0,src:e.url,autoPlay:!0,loop:!0,onLoadedMetadata:w=>c(w.target.duration)}),d&&o&&r.jsxs("div",{className:"video-duration-overlay",children:[r.jsx(Bs,{size:14}),r.jsx("span",{children:Lc(o)})]})]}),r.jsxs("div",{className:"media-info",children:[r.jsxs("div",{className:"media-meta",children:[e.filename||"Generated Video",o&&r.jsxs("span",{className:"duration-inline",children:[" • ",Lc(o)]})]}),r.jsxs("div",{className:"media-actions",children:[e.url&&r.jsx("a",{className:"icon-btn",href:e.url,download:e.filename||void 0,title:"Download",children:r.jsx(vt,{size:18})}),e.backendUrl&&r.jsx("a",{className:"icon-btn",href:e.backendUrl,target:"_blank",rel:"noreferrer",title:"Open in new tab",children:r.jsx(Pc,{size:18})})]})]})]}):e.kind==="image"?r.jsxs("div",{className:"media-container",children:[r.jsx("img",{className:"media-preview",src:e.url,alt:"Generated",onError:w=>{console.error("Image load failed:",e.url),w.target.style.display="none",w.target.parentNode.innerHTML+=`<div style="padding:20px;color:red">Failed to load image: ${e.url}</div>`}}),r.jsxs("div",{className:"media-info",children:[r.jsx("div",{className:"media-meta",children:e.filename||"Generated Image"}),r.jsxs("div",{className:"media-actions",children:[e.url&&r.jsx("a",{className:"icon-btn",href:e.url,download:e.filename||void 0,title:"Download",children:r.jsx(vt,{size:18})}),e.backendUrl&&r.jsx("a",{className:"icon-btn",href:e.backendUrl,target:"_blank",rel:"noreferrer",title:"Open in new tab",children:r.jsx(Pc,{size:18})})]})]})]}):e.kind==="lora"?r.jsxs("div",{className:"media-container",style:{padding:"24px"},children:[r.jsx("h3",{children:"LoRA Training Complete"}),r.jsxs("div",{className:"media-meta",style:{marginTop:"16px"},children:[r.jsxs("p",{children:["ID: ",e.training_id]}),r.jsxs("p",{children:["Path: ",e.lora_path]})]})]}):null:r.jsxs("div",{className:"placeholder-state",children:[r.jsx("div",{className:"placeholder-icon",children:r.jsx(da,{})}),r.jsx("h3",{children:"Ready to Create"}),r.jsx("p",{className:"muted",children:"Configure parameters and click Generate"})]}),[e]);return r.jsxs("section",{className:"output-panel",children:[r.jsxs("div",{style:{position:"absolute",top:20,right:20,zIndex:10,display:"flex",gap:"8px"},children:[r.jsx("button",{className:"icon-btn",onClick:()=>l(!s),title:"History",children:r.jsx(Qh,{size:20})}),a&&r.jsx("button",{className:"icon-btn",onClick:a,title:"Close & show My Media",children:r.jsx(Qe,{size:20})})]}),k,s&&r.jsxs("div",{className:"history",children:[r.jsxs("div",{className:"history-title",style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsx("span",{children:"History"}),r.jsx("button",{className:"icon-btn",onClick:()=>l(!1),children:r.jsx(Qe,{size:18})})]}),r.jsxs("div",{className:"history-list",children:[g&&r.jsx("div",{style:{padding:20,textAlign:"center"},className:"muted",children:"Loading..."}),x&&r.jsx("div",{className:"error",children:x}),!g&&!x&&v.length===0&&r.jsx("div",{style:{padding:20,textAlign:"center"},className:"muted",children:"No history yet"}),v.map(w=>r.jsxs("button",{className:"history-item",onClick:()=>{n({kind:"video",url:`${oe}/outputs/${w.filename}`,backendUrl:`${oe}/outputs/${w.filename}`,filename:w.filename})},children:[r.jsx("div",{className:"history-item-title",children:w.filename}),r.jsx("div",{className:"history-item-sub",children:new Date(w.mtime*1e3).toLocaleString()})]},w.filename))]})]})]})}function Gx({onJobComplete:e,refreshToken:t}){const[n,a]=i.useState({running:[],pending:[],total_running:0,total_pending:0}),[s,l]=i.useState([]),[o,c]=i.useState(!1),[d,p]=i.useState(new Set),v=i.useRef(null),g=i.useCallback(async()=>{try{const F=await fetch(`${oe}/comfyui/queue`);if(!F.ok)return;const f=await F.json();a(f)}catch{}},[]),x=i.useCallback(async F=>{try{const f=await fetch(`${oe}/comfyui/job/${F}`);return f.ok?await f.json():null}catch{return null}},[]);i.useEffect(()=>{g();const F=setInterval(g,3e3);return()=>clearInterval(F)},[g,t]),i.useEffect(()=>{for(const F of s)!d.has(F.prompt_id)&&F.status==="completed"&&F.output_video&&(e&&e(F),p(f=>new Set([...f,F.prompt_id])))},[s,d,e]),i.useEffect(()=>{const F=async()=>{for(const f of n.running){const u=await x(f.prompt_id);u&&u.status==="completed"&&l(h=>h.some(y=>y.prompt_id===u.prompt_id)?h:[...h,u].slice(-10))}};n.running.length>0&&F()},[n.running,x]),i.useEffect(()=>{const F=f=>{v.current&&!v.current.contains(f.target)&&c(!1)};if(o)return document.addEventListener("mousedown",F),()=>document.removeEventListener("mousedown",F)},[o]);const k=async F=>{try{await fetch(`${oe}/comfyui/queue/${F}`,{method:"DELETE"}),g()}catch(f){console.error("Failed to cancel job:",f)}},w=n.total_running>0,z=n.total_running+n.total_pending;return r.jsxs("div",{style:{position:"relative"},ref:v,children:[r.jsxs("button",{onClick:()=>c(!o),style:{display:"flex",alignItems:"center",gap:"6px",padding:"6px 10px",backgroundColor:w?"rgba(34, 197, 94, 0.15)":"transparent",border:`1px solid ${w?"#22c55e":"var(--border-color)"}`,borderRadius:"6px",cursor:"pointer",color:"var(--text-primary)",fontSize:"0.8rem"},title:w?`${n.total_running} running, ${n.total_pending} queued`:"No active jobs",children:[w?r.jsx(Oe,{size:14,color:"#22c55e",className:"spin"}):r.jsx(Bs,{size:14,color:"var(--text-muted)"}),r.jsx("span",{style:{fontWeight:500},children:w?n.total_running:0}),n.total_pending>0&&r.jsxs("span",{style:{color:"var(--text-muted)"},children:["+",n.total_pending]})]}),o&&r.jsxs("div",{style:{position:"absolute",top:"100%",right:0,marginTop:"8px",width:"320px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"8px",boxShadow:"0 4px 20px rgba(0,0,0,0.3)",zIndex:1e3,overflow:"hidden"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"10px 12px",borderBottom:"1px solid var(--border-color)",backgroundColor:"var(--bg-primary)"},children:[r.jsx("span",{style:{fontWeight:600,fontSize:"0.85rem"},children:"Generation Queue"}),r.jsxs("div",{style:{display:"flex",gap:"8px"},children:[r.jsx("button",{onClick:g,style:{background:"transparent",border:"none",cursor:"pointer",padding:"2px"},children:r.jsx(mn,{size:12,color:"var(--text-muted)"})}),r.jsx("button",{onClick:()=>c(!1),style:{background:"transparent",border:"none",cursor:"pointer",padding:"2px"},children:r.jsx(Qe,{size:14,color:"var(--text-muted)"})})]})]}),r.jsxs("div",{style:{maxHeight:"300px",overflowY:"auto",padding:"8px"},children:[n.running.length>0&&r.jsxs("div",{style:{marginBottom:"8px"},children:[r.jsx("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginBottom:"4px",textTransform:"uppercase"},children:"Running"}),n.running.map(F=>r.jsx(bl,{job:F,status:"running",onCancel:k},F.prompt_id))]}),n.pending.length>0&&r.jsxs("div",{style:{marginBottom:"8px"},children:[r.jsx("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginBottom:"4px",textTransform:"uppercase"},children:"Pending"}),n.pending.map(F=>r.jsx(bl,{job:F,status:"pending",onCancel:k},F.prompt_id))]}),s.length>0&&r.jsxs("div",{children:[r.jsx("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginBottom:"4px",textTransform:"uppercase"},children:"Completed"}),s.slice(-3).reverse().map(F=>r.jsx(bl,{job:F,status:"completed"},F.prompt_id))]}),z===0&&s.length===0&&r.jsx("div",{style:{textAlign:"center",padding:"16px",color:"var(--text-muted)",fontSize:"0.8rem"},children:"No active jobs"})]})]})]})}function bl({job:e,status:t,onCancel:n}){const a={running:"#22c55e",pending:"#fbbf24",completed:"#3b82f6"},s={running:Oe,pending:Bs,completed:xh}[t];return r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px",padding:"6px 8px",backgroundColor:"var(--bg-input)",borderRadius:"4px",marginBottom:"4px",fontSize:"0.8rem"},children:[r.jsx(s,{size:12,color:a[t],className:t==="running"?"spin":""}),r.jsxs("div",{style:{flex:1,minWidth:0},children:[r.jsx("div",{style:{whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis",fontWeight:500},children:e.prompt||e.prompt_id.slice(0,8)}),r.jsxs("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)"},children:[e.resolution," ",e.aspect_ratio," ",e.num_frames&&`• ${e.num_frames}f`]})]}),t!=="completed"&&n&&r.jsx("button",{onClick:()=>n(e.prompt_id),style:{background:"transparent",border:"none",cursor:"pointer",padding:"2px"},children:r.jsx(Qe,{size:12,color:"var(--text-muted)"})}),t==="completed"&&e.output_video&&r.jsx("a",{href:`${oe}${e.output_video}`,target:"_blank",rel:"noopener noreferrer",style:{color:"#3b82f6",fontSize:"0.7rem"},children:"View"})]})}const gp=i.createContext({nsfwEnabled:!1,setNsfwEnabled:()=>{}}),Dc="oelala_nsfw_enabled";function Qx({children:e}){const[t,n]=i.useState(()=>{try{return localStorage.getItem(Dc)==="true"}catch{return!1}});return i.useEffect(()=>{try{localStorage.setItem(Dc,t.toString())}catch{}},[t]),r.jsx(gp.Provider,{value:{nsfwEnabled:t,setNsfwEnabled:n},children:e})}function vi(){return i.useContext(gp)}function qx({health:e,onRestartBackend:t,restarting:n}){var l,o;const{nsfwEnabled:a,setNsfwEnabled:s}=vi();return r.jsxs("header",{className:"dashboard-header",children:[r.jsx("div",{className:"header-left",children:r.jsxs("h1",{className:"header-title",children:[r.jsx("span",{className:"header-logo",children:"🎬"}),"oelala"]})}),r.jsx("div",{className:"header-center",children:e&&r.jsxs("div",{className:"health-indicator",children:[r.jsx(rh,{size:14,className:e.comfyui?"text-green":"text-red"}),r.jsxs("span",{className:"health-text",children:["ComfyUI: ",e.comfyui?"Online":"Offline"]}),e.gpu_info&&r.jsxs(r.Fragment,{children:[r.jsx(Nh,{size:14,className:"text-purple"}),r.jsxs("span",{className:"health-text",children:[e.gpu_info.name," (",(l=e.gpu_info.vram_used_gb)==null?void 0:l.toFixed(1),"/",(o=e.gpu_info.vram_total_gb)==null?void 0:o.toFixed(1)," GB)"]})]})]})}),r.jsx("div",{className:"header-right",children:r.jsx("button",{className:`nsfw-toggle ${a?"nsfw-enabled":"nsfw-disabled"}`,onClick:()=>s(!a),title:a?"NSFW content visible - Click to hide":"NSFW content hidden - Click to show",children:a?r.jsxs(r.Fragment,{children:[r.jsx(Ph,{size:16}),r.jsx("span",{children:"NSFW"})]}):r.jsxs(r.Fragment,{children:[r.jsx(Eh,{size:16}),r.jsx("span",{children:"SFW"})]})})}),r.jsx("style",{children:`
        .dashboard-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 8px 16px;
          background: var(--bg-secondary, #1a1a1a);
          border-bottom: 1px solid var(--border-color, #333);
          height: 48px;
          flex-shrink: 0;
        }
        .header-left {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .header-title {
          font-size: 18px;
          font-weight: 700;
          margin: 0;
          display: flex;
          align-items: center;
          gap: 6px;
          color: var(--text-color, #fff);
        }
        .header-logo {
          font-size: 22px;
        }
        .header-center {
          display: flex;
          align-items: center;
          gap: 16px;
        }
        .health-indicator {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
          color: var(--text-muted, #888);
        }
        .health-text {
          color: var(--text-secondary, #aaa);
        }
        .text-green { color: #22c55e; }
        .text-red { color: #ef4444; }
        .text-purple { color: #a855f7; }
        .header-right {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .nsfw-toggle {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 12px;
          border-radius: 20px;
          font-size: 12px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
          border: 1px solid;
        }
        .nsfw-toggle.nsfw-disabled {
          background: rgba(34, 197, 94, 0.1);
          border-color: rgba(34, 197, 94, 0.3);
          color: #22c55e;
        }
        .nsfw-toggle.nsfw-disabled:hover {
          background: rgba(34, 197, 94, 0.2);
        }
        .nsfw-toggle.nsfw-enabled {
          background: rgba(239, 68, 68, 0.1);
          border-color: rgba(239, 68, 68, 0.3);
          color: #ef4444;
        }
        .nsfw-toggle.nsfw-enabled:hover {
          background: rgba(239, 68, 68, 0.2);
        }
      `})]})}async function We(e,t,n={}){const a=await fetch(e,{method:"POST",body:t,headers:n,credentials:"same-origin"}),s=await a.text();try{const l=s?JSON.parse(s):null;return{ok:a.ok,status:a.status,data:l}}catch{return{ok:a.ok,status:a.status,data:s}}}async function fa(e,t={}){const n=await fetch(e,{method:"POST",body:JSON.stringify(t),headers:{"Content-Type":"application/json"},credentials:"same-origin"}),a=await n.text();try{const s=a?JSON.parse(a):null;return{ok:n.ok,status:n.status,data:s}}catch{return{ok:n.ok,status:n.status,data:a}}}const bo=[{value:"",label:"None",desc:"No camera motion",prefix:""},{value:"static",label:"📷 Static",desc:"Camera stays still",prefix:"static camera shot, "},{value:"pan_left",label:"⬅️ Pan Left",desc:"Camera pans left",prefix:"camera slowly panning left, "},{value:"pan_right",label:"➡️ Pan Right",desc:"Camera pans right",prefix:"camera slowly panning right, "},{value:"tilt_up",label:"⬆️ Tilt Up",desc:"Camera tilts up",prefix:"camera slowly tilting up, "},{value:"tilt_down",label:"⬇️ Tilt Down",desc:"Camera tilts down",prefix:"camera slowly tilting down, "},{value:"zoom_in",label:"🔍 Zoom In",desc:"Camera zooms in",prefix:"camera slowly zooming in, "},{value:"zoom_out",label:"🔭 Zoom Out",desc:"Camera zooms out",prefix:"camera slowly zooming out, "},{value:"dolly_in",label:"🎬 Dolly In",desc:"Camera moves forward",prefix:"camera dollying forward, "},{value:"dolly_out",label:"🎬 Dolly Out",desc:"Camera moves back",prefix:"camera dollying backward, "},{value:"orbit_left",label:"🔄 Orbit Left",desc:"Camera orbits left",prefix:"camera orbiting left around subject, "},{value:"orbit_right",label:"🔄 Orbit Right",desc:"Camera orbits right",prefix:"camera orbiting right around subject, "},{value:"handheld",label:"📹 Handheld",desc:"Slight shake",prefix:"shaky handheld camera, "},{value:"tracking",label:"🏃 Tracking",desc:"Follows subject",prefix:"camera tracking shot following subject, "},{value:"crane_up",label:"🏗️ Crane Up",desc:"Camera rises up",prefix:"crane shot rising up, "},{value:"crane_down",label:"🏗️ Crane Down",desc:"Camera lowers",prefix:"crane shot lowering down, "}];function vp(e){const t=bo.find(n=>n.value===e);return(t==null?void 0:t.prefix)||""}function yp({value:e,onChange:t,style:n={}}){const a=bo.find(s=>s.value===e);return r.jsxs("div",{style:{marginBottom:"12px",...n},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px",marginBottom:"6px"},children:[r.jsx("span",{style:{fontSize:"0.85rem",color:"var(--text-secondary)"},children:"Camera Motion"}),r.jsx("span",{style:{fontSize:"0.7rem",color:"var(--text-muted)"},children:e?a==null?void 0:a.desc:"Optional"})]}),r.jsx("div",{style:{display:"flex",flexWrap:"wrap",gap:"6px"},children:bo.map(s=>r.jsx("button",{onClick:()=>t(s.value===e?"":s.value),type:"button",style:{padding:"6px 10px",borderRadius:"6px",border:e===s.value?"1px solid var(--accent-color)":"1px solid var(--border-color)",background:e===s.value?"rgba(59, 130, 246, 0.2)":"rgba(255,255,255,0.05)",color:e===s.value?"var(--accent-color)":"var(--text-secondary)",fontSize:"0.8rem",cursor:"pointer",transition:"all 0.15s ease"},title:s.desc,children:s.label},s.value))})]})}const Yx=[{value:"480p",label:"480p",desc:"Fast"},{value:"720p",label:"720p",desc:"Balanced"}],Xx=[8,12,16,24],Kx=["16:9","9:16","1:1","4:3","3:4"];function Jx({onOutput:e,onRefreshHistory:t,onJobSubmitted:n}){const[a,s]=i.useState(()=>localStorage.getItem("t2v_prompt")||""),[l,o]=i.useState("blurry, low quality, distorted, ugly"),[c,d]=i.useState(41),[p,v]=i.useState("1:1"),[g,x]=i.useState("480p"),[k,w]=i.useState(16),[z,F]=i.useState(""),[f,u]=i.useState(!1),[h,y]=i.useState(6),[j,I]=i.useState(1),[_,R]=i.useState(-1),[G,W]=i.useState(20),[b,N]=i.useState(6),[L,ee]=i.useState(!1),[T,ne]=i.useState(""),[ae,D]=i.useState(null),U=H=>{s(H),localStorage.setItem("t2v_prompt",H)},q=i.useMemo(()=>a.trim().length>0&&!L,[a,L]),V=async()=>{var Y,M;if(!a.trim()){ne("Prompt is required");return}ee(!0),ne(""),D(null);const Q=vp(z)+a,C=new FormData;C.append("prompt",Q),C.append("num_frames",String(c)),C.append("aspect_ratio",p),C.append("resolution",g),C.append("fps",String(k));try{const m=await We(`${oe}/generate-text`,C);if(!m.ok)throw new Error(((Y=m.data)==null?void 0:Y.detail)||`Generation failed (status ${m.status})`);const A=(M=m.data)==null?void 0:M.prompt_id;if(!A)throw new Error("No prompt_id returned");D({promptId:A,prompt:a.substring(0,40)+(a.length>40?"...":"")}),n&&n({prompt_id:A})}catch(m){const A=(m==null?void 0:m.message)||"Failed to generate video";ne(A),await Ws({level:"error",message:"Text-to-video failed",timestamp:new Date().toISOString(),meta:{message:A}})}finally{ee(!1)}};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(yr,{size:18}),"Video Prompt"]}),r.jsx("textarea",{className:"prompt-textarea",value:a,onChange:H=>U(H.target.value),rows:4,placeholder:"Describe the video you want to generate... (e.g., 'a cat walking through a field of flowers, cinematic')"}),r.jsxs("div",{className:"char-count",children:[a.length," characters"]}),r.jsx(yp,{value:z,onChange:F,style:{marginTop:"12px"}})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Settings"}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Resolution"}),r.jsx("div",{className:"button-group",children:Yx.map(H=>r.jsx("button",{className:`btn-option ${g===H.value?"active":""}`,onClick:()=>x(H.value),type:"button",children:H.label},H.value))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Aspect Ratio"}),r.jsx("div",{className:"button-group",children:Kx.map(H=>r.jsx("button",{className:`btn-option ${p===H?"active":""}`,onClick:()=>v(H),type:"button",children:H},H))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Frame Rate"}),r.jsx("div",{className:"button-group",children:Xx.map(H=>r.jsxs("button",{className:`btn-option ${k===H?"active":""}`,onClick:()=>w(H),type:"button",children:[H," fps"]},H))})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("label",{children:["Duration",r.jsxs("span",{className:"label-value",children:[(c/k).toFixed(1),"s (",c," frames)"]})]}),r.jsx("input",{type:"range",min:"17",max:"81",step:"4",value:c,onChange:H=>d(parseInt(H.target.value,10))}),r.jsxs("div",{className:"range-labels",children:[r.jsxs("span",{children:[(17/k).toFixed(1),"s"]}),r.jsxs("span",{children:[(81/k).toFixed(1),"s"]})]})]})]}),r.jsxs("div",{className:"tool-section collapsible",children:[r.jsxs("button",{className:"section-toggle",onClick:()=>u(!f),children:[r.jsx(vr,{size:16}),"Advanced Settings",r.jsx(Tt,{size:16,className:f?"rotated":""})]}),f&&r.jsxs("div",{className:"advanced-content",children:[r.jsxs("div",{className:"form-row",children:[r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"Video Steps"}),r.jsx("input",{type:"number",value:h,onChange:H=>y(parseInt(H.target.value)||6),min:"1",max:"30"})]}),r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"Video CFG"}),r.jsx("input",{type:"number",value:j,onChange:H=>I(parseFloat(H.target.value)||1),min:"0.1",max:"10",step:"0.1"})]})]}),r.jsxs("div",{className:"form-row",children:[r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"T2I Steps"}),r.jsx("input",{type:"number",value:G,onChange:H=>W(parseInt(H.target.value)||20),min:"1",max:"50"})]}),r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"T2I CFG"}),r.jsx("input",{type:"number",value:b,onChange:H=>N(parseFloat(H.target.value)||6),min:"1",max:"20",step:"0.5"})]})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Seed (-1 = random)"}),r.jsx("input",{type:"number",value:_,onChange:H=>R(parseInt(H.target.value)||-1)})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Negative Prompt"}),r.jsx("textarea",{value:l,onChange:H=>o(H.target.value),rows:2,placeholder:"Things to avoid..."})]})]})]}),ae&&r.jsx("div",{className:"queued-notice",children:"✅ Job queued! Check the Queue panel for progress."}),T&&r.jsxs("div",{className:"error-message",children:["⚠️ ",T]}),r.jsx("button",{className:"btn-primary btn-large",type:"button",disabled:!q,onClick:V,children:L?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Queueing..."]}):r.jsxs(r.Fragment,{children:[r.jsx(yr,{size:18}),"Generate Video"]})}),r.jsx("div",{className:"tool-info",children:"💡 Text-to-Video first generates an image from your prompt, then animates it using Wan2.2"}),r.jsx("style",{children:`
        .prompt-textarea {
          width: 100%;
          padding: 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-family: inherit;
          font-size: 14px;
          resize: vertical;
        }
        .char-count {
          text-align: right;
          font-size: 12px;
          color: var(--text-muted, #888);
          margin-top: 4px;
        }
        .form-group {
          margin-bottom: 16px;
        }
        .form-group label {
          display: flex;
          justify-content: space-between;
          margin-bottom: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .label-value {
          color: var(--accent-color, #7c3aed);
          font-weight: 500;
        }
        .button-group {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }
        .btn-option {
          padding: 8px 16px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: transparent;
          color: var(--text-color, #fff);
          cursor: pointer;
          transition: all 0.2s;
          font-size: 13px;
        }
        .btn-option:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .btn-option.active {
          background: var(--accent-color, #7c3aed);
          border-color: var(--accent-color, #7c3aed);
        }
        .range-labels {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: var(--text-muted, #888);
          margin-top: 4px;
        }
        .form-row {
          display: flex;
          gap: 16px;
        }
        .form-group.half {
          flex: 1;
        }
        .form-group input[type="number"],
        .form-group textarea {
          width: 100%;
          padding: 8px 12px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 13px;
        }
        .section-toggle {
          display: flex;
          align-items: center;
          gap: 8px;
          width: 100%;
          padding: 12px;
          background: transparent;
          border: 1px solid var(--border-color, #333);
          border-radius: 8px;
          color: var(--text-secondary, #aaa);
          cursor: pointer;
          font-size: 13px;
        }
        .section-toggle:hover {
          border-color: var(--border-color, #555);
        }
        .section-toggle .rotated {
          transform: rotate(180deg);
        }
        .section-toggle svg:last-child {
          margin-left: auto;
          transition: transform 0.2s;
        }
        .advanced-content {
          margin-top: 16px;
          padding-top: 16px;
          border-top: 1px solid var(--border-color, #333);
        }
        .progress-section {
          margin: 16px 0;
        }
        .progress-bar {
          height: 4px;
          background: var(--bg-secondary, #333);
          border-radius: 2px;
          overflow: hidden;
        }
        .progress-fill {
          height: 100%;
          background: var(--accent-color, #7c3aed);
          transition: width 0.3s;
        }
        .progress-status {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .error-message {
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          margin: 12px 0;
        }
        .tool-info {
          margin-top: 16px;
          padding: 12px;
          background: rgba(124, 58, 237, 0.1);
          border-radius: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `})]})}function Zx({onPresetChange:e,onParametersChange:t,currentParameters:n}){var _,R,G,W;const[a,s]=i.useState([]),[l,o]=i.useState(null),[c,d]=i.useState({}),[p,v]=i.useState(!0),[g,x]=i.useState(!0),[k,w]=i.useState(null);i.useEffect(()=>{z()},[]);const z=async()=>{var b;try{x(!0);const N=await fetch(`${oe}/api/presets`);if(!N.ok)throw new Error("Failed to fetch presets");const L=await N.json();if(s(L.presets||[]),((b=L.presets)==null?void 0:b.length)>0){const ee=L.presets[0];o(ee),F(ee)}}catch(N){console.error("Failed to load presets:",N),w(N.message),s(eg())}finally{x(!1)}},F=b=>{if(!(b!=null&&b.parameters))return;const N={};Object.entries(b.parameters).forEach(([L,ee])=>{ee.type!=="image"&&(N[L]=ee.default??ee.value??"")}),d(N),t==null||t(N)},f=b=>{o(b),F(b),e==null||e(b)},u=(b,N,L)=>{let ee=N;L.type==="integer"?ee=parseInt(N,10):L.type==="float"&&(ee=parseFloat(N));const T={...c,[b]:ee};d(T),t==null||t(T)},h=b=>{switch(b){case"ImageToVideo":return r.jsx(da,{size:16});case"TextToVideo":return r.jsx(Gt,{size:16});case"TextToImage":return r.jsx(xp,{size:16});default:return r.jsx(vr,{size:16})}},y=b=>{var N,L,ee,T,ne,ae;return(N=b.name)!=null&&N.toLowerCase().includes("lightning")||(L=b.name)!=null&&L.toLowerCase().includes("fast")?r.jsx("span",{className:"preset-badge fast",children:"⚡ Fast"}):(ee=b.name)!=null&&ee.toLowerCase().includes("quality")||(T=b.name)!=null&&T.toLowerCase().includes("q6")?r.jsx("span",{className:"preset-badge quality",children:"💎 Quality"}):(ne=b.name)!=null&&ne.toLowerCase().includes("nsfw")||(ae=b.name)!=null&&ae.toLowerCase().includes("enhanced")?r.jsx("span",{className:"preset-badge nsfw",children:"🔥 Enhanced"}):null},j=(b,N)=>{var ee;const L=c[b]??N.default??"";return N.type==="image"?null:N.type==="string"?r.jsxs("div",{className:"param-group",children:[r.jsxs("label",{htmlFor:`param-${b}`,children:[N.label||b,N.description&&r.jsx("span",{className:"param-hint",title:N.description,children:"ℹ️"})]}),r.jsx("textarea",{id:`param-${b}`,value:L,onChange:T=>u(b,T.target.value,N),placeholder:N.description,rows:b.includes("prompt")?3:1})]},b):N.type==="integer"&&N.min!==void 0&&N.max!==void 0?r.jsxs("div",{className:"param-group",children:[r.jsxs("label",{htmlFor:`param-${b}`,children:[N.label||b,": ",r.jsx("span",{className:"param-value",children:L}),N.description&&r.jsx("span",{className:"param-hint",title:N.description,children:"ℹ️"})]}),r.jsx("input",{id:`param-${b}`,type:"range",min:N.min,max:N.max,step:N.step||1,value:L,onChange:T=>u(b,T.target.value,N)}),r.jsxs("div",{className:"range-labels",children:[r.jsx("span",{children:N.min}),r.jsx("span",{children:N.max})]})]},b):N.type==="float"&&N.min!==void 0&&N.max!==void 0?r.jsxs("div",{className:"param-group",children:[r.jsxs("label",{htmlFor:`param-${b}`,children:[N.label||b,": ",r.jsx("span",{className:"param-value",children:((ee=L.toFixed)==null?void 0:ee.call(L,2))||L}),N.description&&r.jsx("span",{className:"param-hint",title:N.description,children:"ℹ️"})]}),r.jsx("input",{id:`param-${b}`,type:"range",min:N.min,max:N.max,step:N.step||.1,value:L,onChange:T=>u(b,T.target.value,N)}),r.jsxs("div",{className:"range-labels",children:[r.jsx("span",{children:N.min}),r.jsx("span",{children:N.max})]})]},b):N.type==="integer"||N.type==="float"?r.jsxs("div",{className:"param-group",children:[r.jsxs("label",{htmlFor:`param-${b}`,children:[N.label||b,N.description&&r.jsx("span",{className:"param-hint",title:N.description,children:"ℹ️"})]}),r.jsx("input",{id:`param-${b}`,type:"number",value:L,onChange:T=>u(b,T.target.value,N),step:N.step||(N.type==="float"?.1:1)})]},b):N.type==="boolean"?r.jsx("div",{className:"param-group checkbox",children:r.jsxs("label",{htmlFor:`param-${b}`,children:[r.jsx("input",{id:`param-${b}`,type:"checkbox",checked:!!L,onChange:T=>u(b,T.target.checked,N)}),N.label||b,N.description&&r.jsx("span",{className:"param-hint",title:N.description,children:"ℹ️"})]})},b):null},I=()=>{if(!(l!=null&&l.parameters))return{};const b={prompt:[],generation:[],dimensions:[],other:[]};return Object.entries(l.parameters).forEach(([N,L])=>{L.type!=="image"&&(N.includes("prompt")?b.prompt.push([N,L]):["steps","cfg","seed","frame_rate"].includes(N)?b.generation.push([N,L]):["width","height","num_frames"].includes(N)?b.dimensions.push([N,L]):b.other.push([N,L]))}),b};return g?r.jsxs("div",{className:"preset-selector loading",children:[r.jsx(pa,{className:"spinning",size:24}),r.jsx("span",{children:"Loading presets..."})]}):r.jsxs("div",{className:"preset-selector",children:[r.jsxs("div",{className:"preset-header",onClick:()=>v(!p),children:[r.jsxs("div",{className:"preset-title",children:[r.jsx(pa,{size:20}),r.jsx("span",{children:"Workflow Preset"}),l&&r.jsx("span",{className:"selected-preset-name",children:l.name})]}),p?r.jsx(ph,{size:20}):r.jsx(Tt,{size:20})]}),p&&r.jsxs("div",{className:"preset-content",children:[r.jsx("div",{className:"preset-list",children:a.map(b=>r.jsxs("div",{className:`preset-card ${(l==null?void 0:l.id)===b.id?"selected":""}`,onClick:()=>f(b),children:[r.jsxs("div",{className:"preset-card-header",children:[h(b.category),r.jsx("span",{className:"preset-name",children:b.name}),y(b)]}),r.jsx("p",{className:"preset-description",children:b.description})]},b.id))}),l&&r.jsxs("div",{className:"preset-parameters",children:[r.jsxs("h4",{children:[r.jsx(vr,{size:16})," Parameters"]}),((_=I().prompt)==null?void 0:_.length)>0&&r.jsxs("div",{className:"param-section",children:[r.jsx("h5",{children:"📝 Prompts"}),I().prompt.map(([b,N])=>j(b,N))]}),((R=I().generation)==null?void 0:R.length)>0&&r.jsxs("div",{className:"param-section",children:[r.jsx("h5",{children:"⚙️ Generation"}),r.jsx("div",{className:"param-grid",children:I().generation.map(([b,N])=>j(b,N))})]}),((G=I().dimensions)==null?void 0:G.length)>0&&r.jsxs("div",{className:"param-section",children:[r.jsx("h5",{children:"📐 Dimensions"}),r.jsx("div",{className:"param-grid",children:I().dimensions.map(([b,N])=>j(b,N))})]}),((W=I().other)==null?void 0:W.length)>0&&r.jsxs("div",{className:"param-section",children:[r.jsx("h5",{children:"🔧 Other"}),I().other.map(([b,N])=>j(b,N))]})]})]}),k&&r.jsxs("div",{className:"preset-error",children:["⚠️ ",k," - Using default presets"]})]})}function eg(){return[{id:"wan22_enhanced_q4km",name:"WAN 2.2 Enhanced NSFW FastMove",category:"ImageToVideo",description:"Lightning-fast I2V with NSFW FastMove LoRAs. 4 steps, cfg=1.",parameters:{prompt:{type:"string",default:"motion, smooth camera movement",label:"Prompt"},steps:{type:"integer",default:4,min:2,max:12,label:"Steps"},cfg:{type:"float",default:1,min:1,max:3,step:.1,label:"CFG Scale"},seed:{type:"integer",default:-1,label:"Seed",description:"-1 for random"},width:{type:"integer",default:480,min:256,max:1280,step:16,label:"Width"},height:{type:"integer",default:480,min:256,max:1280,step:16,label:"Height"},num_frames:{type:"integer",default:41,min:17,max:81,step:8,label:"Frames"}}},{id:"wan22_q6_quality",name:"WAN 2.2 Q6 Quality",category:"ImageToVideo",description:"Higher quality 6-bit model with DPM++ scheduler. Best visual quality.",parameters:{prompt:{type:"string",default:"cinematic motion",label:"Prompt"},steps:{type:"integer",default:8,min:4,max:20,label:"Steps"},cfg:{type:"float",default:2.5,min:1,max:5,step:.1,label:"CFG Scale"},seed:{type:"integer",default:-1,label:"Seed"},width:{type:"integer",default:512,min:256,max:1280,step:16,label:"Width"},height:{type:"integer",default:512,min:256,max:1280,step:16,label:"Height"},num_frames:{type:"integer",default:49,min:17,max:97,step:8,label:"Frames"}}}]}const tg=[8,12,16,24],rg=[{value:"wan2.2",label:"🎬 Wan2.2 14B Q6 DisTorch2",desc:"High quality via ComfyUI"}],ng={"480p":{label:"480p",dimensions:{"16:9":"848×480","9:16":"480×848","1:1":"480×480","4:3":"640×480","3:4":"480×640"}},"576p":{label:"576p",dimensions:{"16:9":"1024×576","9:16":"576×1024","1:1":"576×576","4:3":"768×576","3:4":"576×768"}},"720p":{label:"720p",dimensions:{"16:9":"1280×720","9:16":"720×1280","1:1":"720×720","4:3":"960×720","3:4":"720×960"}}},ag=["16:9","9:16","1:1","4:3","3:4"];function sg({onOutput:e,onRefreshHistory:t,onCreationsModeChange:n,onParamsChange:a,onJobSubmitted:s}){var ye,ce,me,at,Zt;const{nsfwEnabled:l}=vi(),o=i.useRef(null),[c,d]=i.useState(null),[p,v]=i.useState(""),[g,x]=i.useState("file"),[k,w]=i.useState(()=>{try{return localStorage.getItem("oelala_last_prompt")||""}catch{return""}}),[z,F]=i.useState("low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, unnatural sand behavior, visual glitches"),[f,u]=i.useState(!1),[h,y]=i.useState(!1),[j,I]=i.useState(6),[_,R]=i.useState("480p"),[G,W]=i.useState("wan2.2"),[b,N]=i.useState("v2"),[L,ee]=i.useState(!1),[T,ne]=i.useState("9:16"),[ae,D]=i.useState(16),[U,q]=i.useState(6),[V,H]=i.useState(1),[Q,C]=i.useState(-1),[Y,M]=i.useState(!1),[m,A]=i.useState(""),[X,P]=i.useState({high_noise:[],low_noise:[],general:[]}),[O,te]=i.useState([]),[K,de]=i.useState(!1),[pe,Te]=i.useState({high_noise:[],low_noise:[],pairs:[]}),[nt,Pt]=i.useState("wan2.2_i2v_high_noise_14B_Q6_K.gguf"),[bt,ya]=i.useState("wan2.2_i2v_low_noise_14B_Q6_K.gguf"),[yn,jn]=i.useState(!1),[bn,ja]=i.useState(!1),[It,ve]=i.useState(1),[wn,ba]=i.useState(!1),[kn,wa]=i.useState(null),[Hs,ka]=i.useState({}),[kr,Sn]=i.useState(!1),[Nn,wt]=i.useState(""),[Gs,Ar]=i.useState(null),Sa=i.useMemo(()=>!!c&&!kr,[c,kr]);i.useEffect(()=>{(async()=>{try{const le=await fetch(`${oe}/loras`);if(le.ok){const ue=await le.json();P(ue)}}catch(le){console.error("Failed to fetch LoRAs:",le)}})()},[]);const ft=i.useMemo(()=>{if(l)return X;const E=ue=>(ue||[]).filter(fe=>!fe.nsfw),le={};return ft.by_category&&Object.keys(ft.by_category).forEach(ue=>{const fe=E(ft.by_category[ue]);fe.length>0&&(le[ue]=fe)}),{high_noise:E(X.high_noise),low_noise:E(X.low_noise),general:E(X.general),loras:E(X.loras),by_category:le}},[X,l]);i.useEffect(()=>{(async()=>{try{const le=await fetch(`${oe}/unet-models`);if(le.ok){const ue=await le.json();Te(ue)}}catch(le){console.error("Failed to fetch Unet models:",le)}})()},[]),i.useEffect(()=>{if(k)try{localStorage.setItem("oelala_last_prompt",k)}catch{}},[k]),i.useEffect(()=>{a&&a({tool:"ImageToVideo",prompt:k,duration:j,resolution:_,modelMode:G,modelVersion:b,aspectRatio:T,fps:ae,steps:U,cfg:V,seed:Q,usePose:L,loraConfigs:O,filename:(c==null?void 0:c.name)||null})},[k,j,_,G,b,T,ae,U,V,Q,L,O,c,a]);const S=i.useCallback(async E=>{Ar(E),wt("");try{const le=`${oe}${E.url}`,fe=await(await fetch(le)).blob(),He=E.filename||E.url.split("/").pop(),st=new File([fe],He,{type:fe.type||"image/png"});d(st),v(le),x("file"),e({kind:"image",url:le,backendUrl:le,filename:He,meta:{source:"my-creations",originalItem:E}})}catch(le){wt("Failed to load selected image"),console.error("Error selecting creation:",le)}},[e]);i.useEffect(()=>(n&&n(g==="creations"&&!c,S),()=>{n&&n(!1,null)}),[g,c,n,S]);const $=async E=>{if(!E)return;d(E),wt(""),Ar(null);const le=URL.createObjectURL(E);v(le);try{const ue=new FormData;ue.append("file",E);const fe=await fetch(`${oe}/extract-metadata`,{method:"POST",body:ue});if(fe.ok){const He=await fe.json();He.prompt&&!k&&w(He.prompt),He.negative_prompt&&z==="low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, unnatural sand behavior, visual glitches"&&F(He.negative_prompt)}}catch{}},Z=()=>{d(null),v(""),Ar(null),o.current&&(o.current.value="")},ie=async()=>{var He,st,yi,ji;if(!c){wt("Image is required");return}Sn(!0),wt("");const E=j*ae,le=new FormData;if(le.append("file",c),le.append("num_frames",String(E)),le.append("resolution",_),le.append("fps",String(ae)),le.append("aspect_ratio",T),!L){const Sr=vp(m)+(k||"Motion, subject moving naturally");le.append("prompt",Sr)}let ue,fe=!0;L?(ue=`${oe}/generate-pose`,fe=!1):(ue=`${oe}/generate-wan22-async`,le.append("steps",String(U)),le.append("cfg",String(V)),le.append("seed",String(Q)),bn&&It>1&&(le.append("extend_mode","true"),le.append("clip_count",String(It))),nt&&le.append("unet_high_noise",nt),bt&&le.append("unet_low_noise",bt),O.length>0&&le.append("lora_configs",JSON.stringify(O)));try{const lt=await We(ue,le);if(!lt.ok){wt(((He=lt.data)==null?void 0:He.detail)||`Generation failed (status ${lt.status})`);return}if(fe)s&&s(lt.data);else{const Sr=((st=lt.data)==null?void 0:st.video_url)||((yi=lt.data)==null?void 0:yi.url),wp=(ji=lt.data)==null?void 0:ji.output_video,bi=Sr?`${oe}${Sr}`:"";e({kind:"video",url:bi,backendUrl:bi,filename:wp,meta:lt.data}),t()}}catch(lt){const Sr=(lt==null?void 0:lt.message)||"Failed to generate video";wt(Sr),await Ws({level:"error",message:"Image-to-video failed",timestamp:new Date().toISOString(),meta:{message:Sr,modelMode:G}})}finally{Sn(!1)}};return r.jsxs("div",{className:"tool-container",children:[r.jsx("style",{children:`
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}),r.jsxs("div",{className:"grok-card",children:[r.jsx("div",{className:"grok-card-header",children:r.jsx("div",{className:"grok-card-title",children:"Model Selection"})}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Generation Mode"}),r.jsxs("div",{style:{position:"relative"},children:[r.jsx("select",{value:G,onChange:E=>{W(E.target.value),E.target.value==="wan2.2"&&(R("576p"),ne("9:16"),I(6))},style:{width:"100%",padding:"12px 40px 12px 16px",backgroundColor:"var(--bg-secondary, #1a1a1a)",border:"1px solid var(--border-color)",borderRadius:"8px",color:"var(--text-primary, #fff)",fontSize:"1rem",appearance:"none",cursor:"pointer"},children:rg.map(E=>r.jsx("option",{value:E.value,style:{backgroundColor:"#1a1a1a",color:"#fff"},children:E.label},E.value))}),r.jsx(Tt,{size:20,style:{position:"absolute",right:"12px",top:"50%",transform:"translateY(-50%)",pointerEvents:"none",color:"var(--text-muted)"}})]}),r.jsxs("div",{className:"info-badge",style:{marginTop:"8px"},children:[r.jsx("span",{style:{fontWeight:600},children:"🎬 Wan2.2 14B Q6"})," • ",r.jsx("span",{style:{color:"#93c5fd"},children:"ComfyUI Backend"}),r.jsx("div",{style:{marginTop:"4px",opacity:.8},children:"High-quality I2V with DisTorch2 + SageAttention (10GB VRAM)"})]})]}),r.jsxs("div",{style:{marginTop:"12px",paddingTop:"12px",borderTop:"1px solid var(--border-color)"},children:[r.jsxs("div",{onClick:()=>jn(!yn),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer",padding:"4px 0"},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[r.jsx(fp,{size:16}),r.jsx("span",{style:{fontSize:"0.9rem",fontWeight:500},children:"Unet Model"}),r.jsxs("span",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:["(",nt.replace(".gguf","").replace("wan2.2_i2v_",""),")"]})]}),r.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:yn?"▼":"▶"})]}),yn&&r.jsxs("div",{style:{marginTop:"12px",display:"flex",flexDirection:"column",gap:"12px"},children:[r.jsxs("div",{children:[r.jsx("label",{style:{display:"block",fontSize:"0.8rem",color:"var(--text-muted)",marginBottom:"4px"},children:"Model Pair (recommended)"}),r.jsx("select",{onChange:E=>{var ue;const le=(ue=pe.pairs)==null?void 0:ue.find(fe=>fe.name===E.target.value);le&&(Pt(le.high.path),ya(le.low.path))},style:{width:"100%",padding:"8px 12px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"6px",color:"var(--text-primary)",fontSize:"0.85rem"},value:((ce=(ye=pe.pairs)==null?void 0:ye.find(E=>E.high.path===nt&&E.low.path===bt))==null?void 0:ce.name)||"",children:(me=pe.pairs)==null?void 0:me.map(E=>r.jsxs("option",{value:E.name,children:[E.name," (",E.high.size_gb,"GB)"]},E.name))})]}),r.jsxs("details",{style:{fontSize:"0.8rem"},children:[r.jsx("summary",{style:{cursor:"pointer",color:"var(--text-muted)",marginBottom:"8px"},children:"⚙️ Advanced: Select models separately"}),r.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"12px",paddingTop:"8px"},children:[r.jsxs("div",{children:[r.jsx("label",{style:{display:"block",fontSize:"0.8rem",color:"var(--text-muted)",marginBottom:"4px"},children:"High Noise Model (steps 0-3)"}),r.jsx("select",{value:nt,onChange:E=>Pt(E.target.value),style:{width:"100%",padding:"8px 12px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"6px",color:"var(--text-primary)",fontSize:"0.85rem"},children:(at=pe.high_noise)==null?void 0:at.map(E=>r.jsxs("option",{value:E.path,children:[E.name," (",E.size_gb,"GB)"]},E.path))})]}),r.jsxs("div",{children:[r.jsx("label",{style:{display:"block",fontSize:"0.8rem",color:"var(--text-muted)",marginBottom:"4px"},children:"Low Noise Model (steps 3+)"}),r.jsx("select",{value:bt,onChange:E=>ya(E.target.value),style:{width:"100%",padding:"8px 12px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"6px",color:"var(--text-primary)",fontSize:"0.85rem"},children:(Zt=pe.low_noise)==null?void 0:Zt.map(E=>r.jsxs("option",{value:E.path,children:[E.name," (",E.size_gb,"GB)"]},E.path))})]})]})]})]})]})]}),r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsxs("div",{className:"grok-card-title",style:{display:"flex",alignItems:"center",gap:"6px"},children:["Positive Prompt ",r.jsx("span",{style:{fontWeight:400,color:"var(--text-muted)",fontSize:"0.85rem"},children:"(Describe the motion)"}),r.jsxs("div",{style:{position:"relative",display:"inline-block"},children:[r.jsx("button",{className:"icon-btn",style:{width:"20px",height:"20px",border:"none",background:"transparent",padding:0},onClick:()=>y(!h),title:"Prompt tips",children:r.jsx(ip,{size:14,color:h?"#fbbf24":"#666666"})}),h&&r.jsxs("div",{style:{position:"absolute",top:"100%",left:"50%",transform:"translateX(-50%)",marginTop:"8px",backgroundColor:"#1a1a1a",border:"1px solid #fbbf24",borderRadius:"8px",padding:"12px",width:"280px",zIndex:100,fontSize:"0.8rem",color:"#fbbf24",boxShadow:"0 4px 12px rgba(0,0,0,0.5)"},children:[r.jsx("div",{style:{fontWeight:600,marginBottom:"8px"},children:"💡 Prompt Tips"}),r.jsxs("ul",{style:{margin:0,paddingLeft:"16px",lineHeight:1.6},children:[r.jsx("li",{children:"Structure: [subject + motion] + [scene] + [camera]"}),r.jsx("li",{children:'Focus on motion - "walking slowly", "hair blowing"'}),r.jsx("li",{children:'Add intensity - "quickly", "gently", "dramatically"'}),r.jsx("li",{children:'Camera moves - "slow zoom in", "pan left"'}),r.jsx("li",{children:"Describe what you want, not what to avoid"})]})]})]})]}),r.jsxs("div",{style:{display:"flex",gap:"4px"},children:[r.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px"},onClick:async()=>{if(p)try{const le=await(await fetch(`${oe}/extract-metadata-url`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image_url:p})})).json();le.positive_prompt&&w(le.positive_prompt),le.negative_prompt&&setNegPrompt(le.negative_prompt)}catch(E){console.error("Extract metadata failed:",E)}},title:"Extract prompt from selected image",disabled:!p,children:r.jsx(Fh,{size:14,color:p?"#fbbf24":"#666666"})}),r.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px"},children:r.jsx(mp,{size:14,color:"#fbbf24"})}),r.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px"},children:r.jsx(Gt,{size:14,color:"#fbbf24"})})]})]}),r.jsx(yp,{value:m,onChange:A}),r.jsxs("div",{style:{position:"relative"},children:[r.jsx("textarea",{className:"form-textarea",value:k,onChange:E=>w(E.target.value),rows:4,placeholder:"Describe how you want the image to move or animate... (Optional for image-to-video)",style:{backgroundColor:"#0f0f0f",border:"1px solid var(--border-color)",borderRadius:"8px",resize:"vertical",minHeight:"80px",padding:"12px",paddingBottom:"28px",width:"100%",boxSizing:"border-box"}}),r.jsxs("div",{style:{position:"absolute",bottom:"8px",right:"8px",fontSize:"0.7rem",color:"var(--text-muted)"},children:[k.length,"/2048"]})]}),r.jsxs("div",{style:{marginTop:"12px"},children:[r.jsxs("div",{onClick:()=>u(!f),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer",padding:"8px 0"},children:[r.jsx("span",{style:{fontSize:"0.85rem",color:"var(--text-secondary)"},children:"Negative Prompt"}),r.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:f?"▼":"▶"})]}),f&&r.jsxs("div",{style:{position:"relative"},children:[r.jsx("textarea",{className:"form-textarea",value:z,onChange:E=>F(E.target.value),rows:3,placeholder:"Things to avoid in the generation...",style:{backgroundColor:"#0f0f0f",border:"1px solid var(--border-color)",borderRadius:"8px",resize:"vertical",minHeight:"60px",padding:"12px",paddingBottom:"28px",width:"100%",boxSizing:"border-box",fontSize:"0.85rem"}}),r.jsxs("div",{style:{position:"absolute",bottom:"8px",right:"8px",fontSize:"0.7rem",color:"var(--text-muted)"},children:[z.length,"/2048"]})]})]})]}),r.jsxs("div",{className:"grok-card",children:[r.jsx("div",{className:"grok-card-header",children:r.jsx("div",{className:"grok-card-title",children:"Upload Photo"})}),r.jsxs("div",{className:"grok-tabs",children:[r.jsxs("button",{className:`grok-tab ${g==="file"?"active":""}`,onClick:()=>x("file"),children:[r.jsx(Ye,{size:14})," Upload File"]}),r.jsxs("button",{className:`grok-tab ${g==="url"?"active":""}`,onClick:()=>x("url"),children:[r.jsx(dp,{size:14})," From URL"]}),r.jsxs("button",{className:`grok-tab ${g==="creations"?"active":""}`,onClick:()=>x("creations"),children:[r.jsx(cp,{size:14})," From My Creations"]})]}),r.jsx("input",{ref:o,type:"file",accept:"image/*",onChange:E=>{var le;return $((le=E.target.files)==null?void 0:le[0])},style:{display:"none"}}),g==="file"&&!c&&r.jsxs("div",{className:"upload-box",onClick:()=>{var E;return(E=o.current)==null?void 0:E.click()},style:{cursor:"pointer",borderStyle:"dashed",minHeight:"200px",justifyContent:"center"},children:[r.jsx(Ye,{size:48,className:"text-muted",style:{opacity:.2}}),r.jsx("div",{style:{fontSize:"1rem",fontWeight:500,color:"var(--text-secondary)"},children:"Drag & drop an image here, or click to browse"}),r.jsx("div",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:"JPEG, PNG, WebP, Max 20MB"}),r.jsx("div",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:"Minimum size: 300x300px"})]}),g==="url"&&!c&&r.jsxs("div",{style:{padding:"16px 0"},children:[r.jsx("div",{style:{fontSize:"0.85rem",color:"var(--text-muted)",marginBottom:"8px"},children:"Enter image URL:"}),r.jsx("input",{type:"url",placeholder:"https://example.com/image.jpg",style:{width:"100%",padding:"12px",background:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"8px",color:"var(--text-primary)",fontSize:"0.9rem"},onKeyDown:async E=>{if(E.key==="Enter"&&E.target.value)try{const ue=await(await fetch(E.target.value)).blob(),fe=E.target.value.split("/").pop()||"image.jpg",He=new File([ue],fe,{type:ue.type});$(He)}catch{wt("Failed to load image from URL")}}}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"8px"},children:"Press Enter to load"})]}),g==="creations"&&!c&&r.jsxs("div",{style:{padding:"24px 16px",textAlign:"center",color:"var(--text-muted)",backgroundColor:"var(--bg-secondary)",borderRadius:"8px",border:"1px dashed var(--border-color)"},children:[r.jsx(gr,{size:32,style:{opacity:.5,marginBottom:"12px"}}),r.jsx("div",{style:{fontSize:"0.9rem",marginBottom:"8px"},children:"Select an image from the panel on the right →"}),r.jsx("div",{style:{fontSize:"0.8rem",opacity:.7},children:"Browse your generated images"})]}),c&&r.jsxs("div",{className:"relative",style:{position:"relative"},children:[r.jsx("img",{src:p,alt:"Preview",style:{width:"100%",maxHeight:"400px",objectFit:"contain",borderRadius:"8px",border:"1px solid var(--border-color)"}}),r.jsx("button",{onClick:E=>{E.stopPropagation(),Z()},style:{position:"absolute",top:"12px",right:"12px",background:"rgba(0,0,0,0.7)",border:"none",color:"white",borderRadius:"50%",width:"32px",height:"32px",display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",backdropFilter:"blur(4px)"},children:r.jsx(Qe,{size:18})})]})]}),r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"form-group",children:[r.jsxs("label",{className:"grok-section-label",children:["Resolution",r.jsx("span",{className:"text-muted",style:{fontWeight:400},children:" (Higher = Better Quality, more VRAM)"})]}),r.jsx("div",{className:"grok-toggle-group",children:Object.entries(ng).map(([E,le])=>r.jsxs("button",{className:`grok-toggle-btn ${_===E?"active":""}`,onClick:()=>R(E),children:[le.label,r.jsx("span",{style:{fontSize:"0.7rem",opacity:.7,display:"block"},children:le.dimensions[T]||le.dimensions["1:1"]})]},E))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Aspect Ratio"}),r.jsx("div",{className:"grok-toggle-group",children:ag.map(E=>r.jsx("button",{className:`grok-toggle-btn ${T===E?"active":""}`,onClick:()=>ne(E),children:E},E))})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"8px"},children:[r.jsx("label",{className:"grok-section-label",children:"Duration"}),r.jsxs("span",{className:"nav-badge",style:{fontSize:"0.8rem"},children:[j,"s (",j*ae,"f)"]})]}),r.jsxs("div",{style:{position:"relative",height:"24px",marginBottom:"8px"},children:[r.jsx("input",{type:"range",min:"3",max:"15",step:"1",value:j,onChange:E=>I(parseInt(E.target.value,10)),style:{width:"100%",opacity:0,position:"absolute",zIndex:2,cursor:"pointer"}}),r.jsx("div",{style:{position:"absolute",top:"10px",left:0,right:0,height:"4px",backgroundColor:"#333",borderRadius:"2px"},children:r.jsx("div",{style:{width:`${(j-3)/12*100}%`,height:"100%",backgroundColor:"var(--accent-color, #a855f7)",borderRadius:"2px"}})}),r.jsx("div",{style:{position:"absolute",top:"2px",left:`calc(${(j-3)/12*100}% - 10px)`,width:"20px",height:"20px",backgroundColor:"white",borderRadius:"50%",boxShadow:"0 2px 4px rgba(0,0,0,0.3)"}})]}),r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.75rem",color:"var(--text-muted)"},children:[r.jsx("span",{children:"3s"}),r.jsx("span",{children:"6s (rec)"}),r.jsx("span",{children:"15s"})]})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"8px"},children:[r.jsx("label",{className:"grok-section-label",children:"Frame Rate (FPS)"}),r.jsxs("span",{className:"nav-badge",style:{fontSize:"0.8rem"},children:[ae," fps"]})]}),r.jsx("div",{className:"grok-toggle-group",children:tg.map(E=>r.jsx("button",{className:`grok-toggle-btn ${ae===E?"active":""}`,onClick:()=>D(E),type:"button",children:E},E))}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"8px"},children:"Higher FPS = smoother motion, more VRAM required"})]}),G!=="wan2.2"&&r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Model Version"}),r.jsxs("div",{className:"grok-toggle-group",children:[r.jsx("button",{className:`grok-toggle-btn ${b==="v1"?"active":""}`,onClick:()=>N("v1"),children:"V1"}),r.jsx("button",{className:`grok-toggle-btn ${b==="v2"?"active":""}`,onClick:()=>N("v2"),children:"V2 (Enhanced)"})]}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"8px"},children:"V2 features improved video quality, motion, and optional audio generation"})]}),G==="wan2.2"&&r.jsxs("div",{style:{backgroundColor:"var(--bg-tertiary)",padding:"16px",borderRadius:"8px",marginTop:"8px"},children:[r.jsxs("div",{onClick:()=>ba(!wn),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer"},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[r.jsx(pa,{size:16}),r.jsx("span",{style:{fontWeight:600,fontSize:"0.9rem"},children:"Workflow Presets"}),kn&&r.jsx("span",{style:{fontSize:"0.7rem",backgroundColor:"var(--accent-color)",color:"white",padding:"2px 6px",borderRadius:"4px",marginLeft:"4px"},children:kn.name})]}),r.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:wn?"▼":"▶"})]}),wn&&r.jsx("div",{style:{marginTop:"12px"},children:r.jsx(Zx,{onPresetChange:E=>{var le,ue,fe,He;if(wa(E),E!=null&&E.parameters){const st=E.parameters;(le=st.steps)!=null&&le.default&&q(st.steps.default),(ue=st.cfg)!=null&&ue.default&&H(st.cfg.default),((fe=st.seed)==null?void 0:fe.default)!==void 0&&C(st.seed.default),(He=st.frame_rate)!=null&&He.default&&D(st.frame_rate.default)}},onParametersChange:E=>{ka(E),E.steps!==void 0&&q(E.steps),E.cfg!==void 0&&H(E.cfg),E.seed!==void 0&&C(E.seed),E.frame_rate!==void 0&&D(E.frame_rate)},currentParameters:{steps:U,cfg:V,seed:Q,frame_rate:ae}})})]}),G==="wan2.2"&&r.jsxs("div",{style:{backgroundColor:"var(--bg-tertiary)",padding:"16px",borderRadius:"8px",marginTop:"8px"},children:[r.jsxs("div",{onClick:()=>M(!Y),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer"},children:[r.jsx("div",{style:{fontSize:"0.9rem",fontWeight:600,color:"var(--text-primary)"},children:"⚙️ Sampling Settings"}),r.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:Y?"▼":"▶"})]}),Y&&r.jsxs("div",{style:{marginTop:"12px"},children:[r.jsxs("div",{className:"form-group",style:{marginBottom:"12px"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"Sampling Steps"}),r.jsx("span",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:U})]}),r.jsx("input",{type:"range",min:"4",max:"20",step:"1",value:U,onChange:E=>q(parseInt(E.target.value,10)),style:{width:"100%",cursor:"pointer"}}),r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.7rem",color:"var(--text-muted)"},children:[r.jsx("span",{children:"4 (fast)"}),r.jsx("span",{children:"6 (rec)"}),r.jsx("span",{children:"20 (quality)"})]})]}),r.jsxs("div",{className:"form-group",style:{marginBottom:"12px"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"CFG Guidance"}),r.jsx("span",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:V.toFixed(1)})]}),r.jsx("input",{type:"range",min:"1.0",max:"10.0",step:"0.5",value:V,onChange:E=>H(parseFloat(E.target.value)),style:{width:"100%",cursor:"pointer"}}),r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.7rem",color:"var(--text-muted)"},children:[r.jsx("span",{children:"1.0 (rec)"}),r.jsx("span",{children:"5.0"}),r.jsx("span",{children:"10.0"})]})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Seed"}),r.jsxs("div",{style:{display:"flex",gap:"8px"},children:[r.jsx("input",{type:"number",value:Q,onChange:E=>C(parseInt(E.target.value,10)),placeholder:"-1 for random",style:{flex:1,padding:"8px 12px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"6px",color:"var(--text-primary)",fontSize:"0.9rem"}}),r.jsx("button",{className:"btn ghost sm",onClick:()=>C(-1),style:{whiteSpace:"nowrap"},children:"Random"})]}),r.jsx("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginTop:"4px"},children:"-1 = random seed each generation"})]})]}),r.jsxs("div",{style:{marginTop:"16px",paddingTop:"16px",borderTop:"1px solid var(--border-color)"},children:[r.jsxs("div",{onClick:()=>de(!K),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer",marginBottom:K?"12px":0},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[r.jsx(xi,{size:16}),r.jsx("span",{style:{fontWeight:600,fontSize:"0.9rem"},children:"LoRA Models"}),O.length>0&&r.jsxs("span",{style:{fontSize:"0.7rem",backgroundColor:"var(--accent-color)",color:"white",padding:"2px 6px",borderRadius:"4px"},children:[O.length," active"]})]}),r.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:K?"▼":"▶"})]}),K&&r.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"12px"},children:[O.map((E,le)=>r.jsxs("div",{style:{backgroundColor:"var(--bg-input)",borderRadius:"8px",padding:"12px",border:"1px solid var(--border-color)"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"8px"},children:[r.jsxs("span",{style:{fontSize:"0.8rem",fontWeight:600},children:["LoRA #",le+1]}),r.jsx("button",{onClick:()=>te(O.filter((ue,fe)=>fe!==le)),style:{background:"transparent",border:"none",color:"#ef4444",cursor:"pointer",padding:"2px 6px",fontSize:"0.8rem"},children:"✕ Remove"})]}),r.jsxs("div",{style:{marginBottom:"8px"},children:[r.jsx("label",{style:{display:"block",fontSize:"0.75rem",color:"var(--text-muted)",marginBottom:"4px"},children:"High Noise (steps 0-3)"}),r.jsxs("select",{value:E.high||"",onChange:ue=>{const fe=[...O];fe[le]={...E,high:ue.target.value},te(fe)},style:{width:"100%",padding:"6px 10px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"4px",color:"var(--text-primary)",fontSize:"0.8rem"},children:[r.jsx("option",{value:"",children:"None"}),ft.by_category&&Object.keys(ft.by_category).sort().map(ue=>r.jsx("optgroup",{label:ue==="root"?"📁 Root":`📁 ${ue}`,children:ft.by_category[ue].map(fe=>r.jsxs("option",{value:fe.path,children:[fe.name," (",fe.size_mb,"MB)"]},fe.path))},ue))]})]}),r.jsxs("div",{style:{marginBottom:"8px"},children:[r.jsx("label",{style:{display:"block",fontSize:"0.75rem",color:"var(--text-muted)",marginBottom:"4px"},children:"Low Noise (steps 3+)"}),r.jsxs("select",{value:E.low||"",onChange:ue=>{const fe=[...O];fe[le]={...E,low:ue.target.value},te(fe)},style:{width:"100%",padding:"6px 10px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"4px",color:"var(--text-primary)",fontSize:"0.8rem"},children:[r.jsx("option",{value:"",children:"None (uses High Noise)"}),ft.by_category&&Object.keys(ft.by_category).sort().map(ue=>r.jsx("optgroup",{label:ue==="root"?"📁 Root":`📁 ${ue}`,children:ft.by_category[ue].map(fe=>r.jsxs("option",{value:fe.path,children:[fe.name," (",fe.size_mb,"MB)"]},fe.path))},ue))]})]}),r.jsxs("div",{children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"2px"},children:[r.jsx("label",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Strength"}),r.jsx("span",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:(E.strength||1).toFixed(2)})]}),r.jsx("input",{type:"range",min:"0",max:"2",step:"0.05",value:E.strength||1,onChange:ue=>{const fe=[...O];fe[le]={...E,strength:parseFloat(ue.target.value)},te(fe)},style:{width:"100%",cursor:"pointer"}})]})]},le)),r.jsx("button",{onClick:()=>te([...O,{high:"",low:"",strength:1}]),style:{padding:"8px 12px",backgroundColor:"transparent",border:"1px dashed var(--border-color)",borderRadius:"6px",color:"var(--text-secondary)",cursor:"pointer",fontSize:"0.85rem",display:"flex",alignItems:"center",justifyContent:"center",gap:"6px"},children:"+ Add LoRA"}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",fontStyle:"italic"},children:"💡 Stack multiple LoRAs for combined effects. Each LoRA has its own strength."})]})]})]}),r.jsxs("div",{className:"form-group",style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsxs("div",{children:[r.jsx("div",{className:"grok-section-label",style:{marginBottom:"4px"},children:"Generate Audio"}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Enable audio generation (increases credits)"})]}),r.jsxs("label",{className:"grok-switch",children:[r.jsx("input",{type:"checkbox"}),r.jsx("span",{className:"grok-slider"})]})]}),r.jsxs("div",{className:"form-group",style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsxs("div",{children:[r.jsx("div",{className:"grok-section-label",style:{marginBottom:"4px"},children:"Camera Fixed"}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Whether to fix the camera position"})]}),r.jsxs("label",{className:"grok-switch",children:[r.jsx("input",{type:"checkbox"}),r.jsx("span",{className:"grok-slider"})]})]}),r.jsxs("div",{className:"form-group",style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsxs("div",{children:[r.jsx("div",{className:"grok-section-label",style:{marginBottom:"4px"},children:"🎬 Extend Duration"}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Chain multiple clips sequentially"})]}),r.jsxs("label",{className:"grok-switch",children:[r.jsx("input",{type:"checkbox",checked:bn,onChange:E=>{ja(E.target.checked),E.target.checked||ve(1)}}),r.jsx("span",{className:"grok-slider"})]})]}),bn&&r.jsxs("div",{className:"form-group",style:{background:"linear-gradient(135deg, rgba(233, 69, 96, 0.1) 0%, rgba(233, 69, 96, 0.05) 100%)",borderRadius:"8px",padding:"12px",marginTop:"-8px",border:"1px solid rgba(233, 69, 96, 0.2)"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"8px"},children:[r.jsxs("div",{className:"grok-section-label",children:["Number of Clips: ",It]}),r.jsxs("div",{style:{fontSize:"0.75rem",color:"#e94560",background:"rgba(233, 69, 96, 0.15)",padding:"2px 8px",borderRadius:"10px",fontWeight:"600"},children:["≈ ",(j*It).toFixed(0),"s total"]})]}),r.jsx("input",{type:"range",min:"1",max:"5",value:It,onChange:E=>ve(parseInt(E.target.value)),style:{width:"100%",accentColor:"#e94560"}}),r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.7rem",color:"var(--text-muted)",marginTop:"4px"},children:[r.jsx("span",{children:"1"}),r.jsx("span",{children:"2"}),r.jsx("span",{children:"3"}),r.jsx("span",{children:"4"}),r.jsx("span",{children:"5"})]}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"8px",fontStyle:"italic"},children:"🔗 Each clip continues from the last frame of the previous clip"})]})]}),r.jsxs("div",{className:"grok-card",children:[r.jsx("div",{className:"grok-card-header",children:r.jsx("div",{className:"grok-card-title",children:"Aspect Ratio"})}),r.jsx("div",{className:"aspect-grid",children:[{label:"Auto",icon:r.jsx(Gt,{size:16})},{label:"21:9",icon:r.jsx("div",{style:{width:"24px",height:"10px",border:"1px solid currentColor"}})},{label:"16:9",icon:r.jsx("div",{style:{width:"24px",height:"14px",border:"1px solid currentColor"}})},{label:"4:3",icon:r.jsx("div",{style:{width:"20px",height:"15px",border:"1px solid currentColor"}})},{label:"1:1",icon:r.jsx("div",{style:{width:"18px",height:"18px",border:"1px solid currentColor"}})},{label:"3:4",icon:r.jsx("div",{style:{width:"15px",height:"20px",border:"1px solid currentColor"}})},{label:"9:16",icon:r.jsx("div",{style:{width:"14px",height:"24px",border:"1px solid currentColor"}})}].map(E=>r.jsxs("button",{className:`aspect-btn ${T===E.label?"active":""}`,onClick:()=>ne(E.label),children:[r.jsx("div",{className:"aspect-icon",style:{background:"transparent",display:"flex",alignItems:"center",justifyContent:"center",border:"none"},children:E.icon}),r.jsx("span",{className:"aspect-label",children:E.label})]},E.label))})]}),Nn&&r.jsx("div",{style:{padding:"12px",backgroundColor:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.2)",borderRadius:"8px",color:"#ef4444",marginBottom:"16px",fontSize:"0.9rem"},children:Nn}),r.jsx("button",{className:"primary-btn",disabled:!Sa,onClick:ie,style:{height:"48px",fontSize:"1rem",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px",backgroundColor:"#e5e5e5",color:"black"},children:kr?r.jsx(r.Fragment,{children:"Generating..."}):r.jsxs(r.Fragment,{children:[r.jsx(Gt,{size:18}),"Generate from Image"]})}),kr&&r.jsx("div",{className:"progress-container",children:r.jsx("div",{className:"progress-indeterminate"})})]})}const Mt={wan22:[{value:"wan2.2-t2i",label:"Wan2.2 T2I (Multi-GPU)",category:"Video Model"}],flux:[{value:"flux1-dev-fp8",label:"Flux.1 Dev (FP8)",category:"Flux"}],sdxl:[{value:"CyberRealistic_Pony_v14.1_FP16.safetensors",label:"CyberRealistic Pony",category:"Realistic/Pony"},{value:"dreamshaperXL_lightningDPMSDE.safetensors",label:"Dreamshaper Lightning",category:"General"},{value:"illustriousRealismBy_v10VAE.safetensors",label:"Illustrious Realism",category:"Realistic"},{value:"juggernautXL_ragnarok.safetensors",label:"Juggernaut XL",category:"General"},{value:"novaAnimeXL_ilV150.safetensors",label:"Nova Anime XL",category:"Anime"},{value:"ponyDiffusionV6XL_v6StartWithThisOne.safetensors",label:"Pony Diffusion V6",category:"Pony"},{value:"reapony_v90.safetensors",label:"Reapony V9",category:"Realistic/Pony"},{value:"ultraRealisticByStable_v20FP16.safetensors",label:"Ultra Realistic",category:"Realistic"},{value:"waiIllustriousSDXL_v160.safetensors",label:"Wai Illustrious",category:"Anime"}],sd15:[{value:"Realistic_Vision_V5.1.safetensors",label:"Realistic Vision V5.1",category:"Realistic"}],diffusers:[{value:"sd3.5-large-int8",label:"SD3.5 Large (INT8)"},{value:"realvisxl-v5.0",label:"RealVisXL V5.0"}]},At=e=>e==="wan2.2-t2i"?"wan22":e.startsWith("flux")?"flux":e==="Realistic_Vision_V5.1.safetensors"?"sd15":e.endsWith(".safetensors")?"sdxl":"diffusers";function lg({onOutput:e,onJobSubmitted:t}){const{nsfwEnabled:n}=vi(),[a,s]=i.useState(""),[l,o]=i.useState("ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text"),[c,d]=i.useState("1:1"),[p,v]=i.useState("normal"),[g,x]=i.useState("CyberRealistic_Pony_v14.1_FP16.safetensors"),[k,w]=i.useState(1),[z,F]=i.useState(!1),[f,u]=i.useState(""),[h,y]=i.useState(!1),[j,I]=i.useState(null),[_,R]=i.useState([]),[G,W]=i.useState([{name:"None",strength:1},{name:"None",strength:1},{name:"None",strength:1}]),[b,N]=i.useState(30),[L,ee]=i.useState(7.5),[T,ne]=i.useState(3.5),[ae,D]=i.useState(-1),[U,q]=i.useState("dpmpp_2m"),[V,H]=i.useState("karras");i.useEffect(()=>{(async()=>{try{const A=await fetch(`${oe}/loras`);if(A.ok){const X=await A.json();R(X.loras||[])}}catch(A){console.warn("Failed to fetch LoRAs:",A)}})()},[]);const Q=i.useMemo(()=>n?_:_.filter(m=>!m.nsfw),[_,n]),C=(m,A,X)=>{W(P=>{const O=[...P];return O[m]={...O[m],[A]:X},O})},Y=async()=>{var m,A,X;if(a.trim()){F(!0),u(""),I(null);try{const P=[];for(let O=0;O<k;O++){const te=`t2i-${Date.now()}-${Math.random().toString(36).slice(2,8)}`,K=new FormData;K.append("prompt",a),K.append("aspect_ratio",c);const de=At(g);let pe="/generate-image";if(de==="wan22")pe="/generate-wan22-t2i",K.append("steps",b),K.append("seed",ae);else if(de==="flux")pe="/generate-flux",K.append("steps",b),K.append("guidance",T),K.append("seed",ae);else if(de==="sdxl"){pe="/generate-sdxl",K.append("checkpoint",g),K.append("negative_prompt",l),K.append("steps",b),K.append("cfg",L),K.append("seed",ae),K.append("sampler_name",U),K.append("scheduler",V);const nt=G.filter(Pt=>Pt.name&&Pt.name!=="None");nt.length>0&&K.append("lora_configs",JSON.stringify(nt))}else de==="sd15"?(pe="/generate-sd15",K.append("negative_prompt",l),K.append("steps",b),K.append("cfg",L),K.append("seed",ae),K.append("sampler_name",U),K.append("scheduler",V)):(K.append("mode",p),K.append("model",g),K.append("job_id",te));const Te=await We(`${oe}${pe}`,K);if(!Te.ok)throw new Error(((m=Te.data)==null?void 0:m.detail)||`Generation failed (status ${Te.status})`);(A=Te.data)!=null&&A.prompt_id&&P.push(Te.data.prompt_id),t&&t({prompt_id:(X=Te.data)==null?void 0:X.prompt_id})}I({count:k,model:M(),promptIds:P})}catch(P){console.error("Generation error:",P),u(P.message||"Failed to generate image")}finally{F(!1)}}},M=()=>{const A=[...Mt.wan22,...Mt.flux,...Mt.sdxl,...Mt.sd15,...Mt.diffusers].find(X=>X.value===g);return(A==null?void 0:A.label)||g};return r.jsxs("div",{className:"tool-container",children:[r.jsx("div",{className:"grok-card",children:r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Mode"}),r.jsxs("div",{className:"form-select",style:{display:"flex",alignItems:"center",gap:"8px",cursor:"pointer"},children:[r.jsx(Gt,{size:16,className:"text-primary"}),r.jsx("span",{children:"Normal"})]}),r.jsxs("div",{className:"info-badge",children:[r.jsx("span",{style:{color:"#93c5fd"},children:"Standard Quality"}),r.jsx("div",{style:{marginTop:"4px",opacity:.8},children:"Fast and efficient image generation (1 credit per image)"})]})]})}),r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Enter Image Prompt"}),r.jsxs("div",{style:{display:"flex",gap:"4px"},children:[r.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px",fontSize:"10px"},children:"T"}),r.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px",fontSize:"10px"},children:"✨"})]})]}),r.jsx("div",{style:{position:"relative"},children:r.jsx("textarea",{className:"form-textarea",value:a,onChange:m=>s(m.target.value),rows:4,placeholder:"A attractive blonde woman with cup f, tattoos, looking at me defiantly.",style:{backgroundColor:"#0f0f0f",border:"none",resize:"none",paddingBottom:"24px"}})})]}),r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Model"}),r.jsx("span",{className:"nav-badge",style:{fontSize:"0.7rem"},children:At(g).toUpperCase()})]}),r.jsxs("div",{style:{marginBottom:"12px"},children:[r.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"⚡ Flux (Best Quality)"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:Mt.flux.map(m=>r.jsx("button",{className:`grok-toggle-btn ${g===m.value?"active":""}`,onClick:()=>x(m.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},children:m.label},m.value))})]}),r.jsxs("div",{style:{marginBottom:"12px"},children:[r.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"🎨 SDXL Checkpoints"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:Mt.sdxl.map(m=>r.jsx("button",{className:`grok-toggle-btn ${g===m.value?"active":""}`,onClick:()=>x(m.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},title:m.category,children:m.label},m.value))})]}),r.jsxs("div",{style:{marginBottom:"12px"},children:[r.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"🖼️ SD 1.5 (Fast, Low VRAM)"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:Mt.sd15.map(m=>r.jsx("button",{className:`grok-toggle-btn ${g===m.value?"active":""}`,onClick:()=>x(m.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},children:m.label},m.value))})]}),r.jsxs("div",{style:{marginBottom:"12px"},children:[r.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"🎬 Wan2.2 (Video Model T2I)"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:Mt.wan22.map(m=>r.jsx("button",{className:`grok-toggle-btn ${g===m.value?"active":""}`,onClick:()=>x(m.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},children:m.label},m.value))})]}),r.jsxs("div",{children:[r.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"🐍 Diffusers (Python)"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:Mt.diffusers.map(m=>r.jsx("button",{className:`grok-toggle-btn ${g===m.value?"active":""}`,onClick:()=>x(m.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},children:m.label},m.value))})]})]}),(At(g)==="sdxl"||At(g)==="sd15")&&r.jsxs("div",{className:"grok-card",children:[r.jsx("div",{className:"grok-card-header",children:r.jsx("div",{className:"grok-card-title",children:"Negative Prompt"})}),r.jsx("textarea",{className:"form-textarea",value:l,onChange:m=>o(m.target.value),rows:2,placeholder:"ugly, deformed, blurry...",style:{backgroundColor:"#0f0f0f",border:"none",resize:"none",fontSize:"0.85rem"}})]}),r.jsxs("div",{className:"grok-card",children:[r.jsx("div",{className:"grok-card-header",children:r.jsx("div",{className:"grok-card-title",children:"Aspect Ratio"})}),r.jsx("div",{className:"aspect-grid",style:{gridTemplateColumns:"repeat(5, 1fr)"},children:[{label:"1:1",icon:r.jsx("div",{style:{width:"18px",height:"18px",border:"1px solid currentColor"}})},{label:"16:9",icon:r.jsx("div",{style:{width:"24px",height:"14px",border:"1px solid currentColor"}})},{label:"9:16",icon:r.jsx("div",{style:{width:"14px",height:"24px",border:"1px solid currentColor"}})},{label:"4:3",icon:r.jsx("div",{style:{width:"20px",height:"15px",border:"1px solid currentColor"}})},{label:"3:4",icon:r.jsx("div",{style:{width:"15px",height:"20px",border:"1px solid currentColor"}})},{label:"2:3",icon:r.jsx("div",{style:{width:"16px",height:"24px",border:"1px solid currentColor"}})},{label:"3:2",icon:r.jsx("div",{style:{width:"24px",height:"16px",border:"1px solid currentColor"}})},{label:"4:5",icon:r.jsx("div",{style:{width:"16px",height:"20px",border:"1px solid currentColor"}})},{label:"5:4",icon:r.jsx("div",{style:{width:"20px",height:"16px",border:"1px solid currentColor"}})},{label:"9:21",icon:r.jsx("div",{style:{width:"10px",height:"24px",border:"1px solid currentColor"}})},{label:"21:9",icon:r.jsx("div",{style:{width:"24px",height:"10px",border:"1px solid currentColor"}})}].map(m=>r.jsxs("button",{className:`aspect-btn ${c===m.label?"active":""}`,onClick:()=>d(m.label),style:{height:"60px"},children:[r.jsx("div",{className:"aspect-icon",style:{background:"transparent",display:"flex",alignItems:"center",justifyContent:"center",border:"none",marginBottom:"4px"},children:m.icon}),r.jsx("span",{className:"aspect-label",style:{fontSize:"0.65rem"},children:m.label})]},m.label))})]}),r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",style:{cursor:"pointer"},onClick:()=>y(!h),children:[r.jsx("div",{className:"grok-card-title",children:"Advanced Settings"}),r.jsx(Tt,{size:16,className:"text-muted",style:{transform:h?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.2s"}})]}),h&&r.jsxs(r.Fragment,{children:[r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Batch Count"}),r.jsx("div",{className:"grok-toggle-group",children:[1,2,3,4].map(m=>r.jsx("button",{className:`grok-toggle-btn ${k===m?"active":""}`,onClick:()=>w(m),children:m},m))})]}),At(g)==="flux"&&r.jsxs(r.Fragment,{children:[r.jsxs("div",{className:"form-group",style:{marginTop:"12px"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"Steps"}),r.jsx("span",{className:"nav-badge",children:b})]}),r.jsx("input",{type:"range",min:"10",max:"30",value:b,onChange:m=>N(parseInt(m.target.value)),className:"form-range"})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"Guidance"}),r.jsx("span",{className:"nav-badge",children:T})]}),r.jsx("input",{type:"range",min:"1",max:"10",step:"0.5",value:T,onChange:m=>ne(parseFloat(m.target.value)),className:"form-range"})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Seed (-1 = random)"}),r.jsx("input",{type:"number",value:ae,onChange:m=>D(parseInt(m.target.value)||-1),className:"form-input",style:{backgroundColor:"#0f0f0f",border:"1px solid #333",borderRadius:"6px",padding:"8px",width:"100%"}})]})]}),At(g)==="wan22"&&r.jsxs(r.Fragment,{children:[r.jsxs("div",{className:"form-group",style:{marginTop:"12px"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"Steps"}),r.jsx("span",{className:"nav-badge",children:b})]}),r.jsx("input",{type:"range",min:"10",max:"50",value:b,onChange:m=>N(parseInt(m.target.value)),className:"form-range"}),r.jsx("div",{style:{fontSize:"0.7rem",opacity:.6,marginTop:"4px"},children:"Multi-GPU workflow (DisTorch2) - 2-stage denoising"})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Seed (-1 = random)"}),r.jsx("input",{type:"number",value:ae,onChange:m=>D(parseInt(m.target.value)||-1),className:"form-input",style:{backgroundColor:"#0f0f0f",border:"1px solid #333",borderRadius:"6px",padding:"8px",width:"100%"}})]})]}),(At(g)==="sdxl"||At(g)==="sd15")&&r.jsxs(r.Fragment,{children:[r.jsxs("div",{className:"form-group",style:{marginTop:"12px"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"Steps"}),r.jsx("span",{className:"nav-badge",children:b})]}),r.jsx("input",{type:"range",min:"10",max:"50",value:b,onChange:m=>N(parseInt(m.target.value)),className:"form-range"})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"CFG Scale"}),r.jsx("span",{className:"nav-badge",children:L})]}),r.jsx("input",{type:"range",min:"1",max:"15",step:"0.5",value:L,onChange:m=>ee(parseFloat(m.target.value)),className:"form-range"})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Sampler"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"4px"},children:["euler","euler_ancestral","dpmpp_2m","dpmpp_sde"].map(m=>r.jsx("button",{className:`grok-toggle-btn ${U===m?"active":""}`,onClick:()=>q(m),style:{fontSize:"0.7rem",padding:"4px 8px"},children:m},m))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Scheduler"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"4px"},children:["normal","karras","exponential","sgm_uniform"].map(m=>r.jsx("button",{className:`grok-toggle-btn ${V===m?"active":""}`,onClick:()=>H(m),style:{fontSize:"0.7rem",padding:"4px 8px"},children:m},m))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Seed (-1 = random)"}),r.jsx("input",{type:"number",value:ae,onChange:m=>D(parseInt(m.target.value)||-1),className:"form-input",style:{backgroundColor:"#0f0f0f",border:"1px solid #333",borderRadius:"6px",padding:"8px",width:"100%"}})]}),At(g)==="sdxl"&&Q.length>0&&r.jsxs("div",{className:"form-group",children:[r.jsxs("label",{className:"grok-section-label",style:{marginBottom:"8px"},children:["LoRAs (up to 3) ",!n&&_.length>Q.length&&r.jsxs("span",{style:{fontSize:"0.65rem",color:"var(--text-muted)",marginLeft:"8px"},children:["(",_.length-Q.length," hidden)"]})]}),G.map((m,A)=>r.jsxs("div",{style:{display:"flex",gap:"8px",marginBottom:"8px",alignItems:"center"},children:[r.jsxs("select",{value:m.name,onChange:X=>C(A,"name",X.target.value),style:{flex:1,backgroundColor:"#0f0f0f",border:"1px solid #333",borderRadius:"6px",padding:"6px 8px",color:"#fff",fontSize:"0.75rem"},children:[r.jsx("option",{value:"None",children:"None"}),Q.map(X=>r.jsx("option",{value:X.name,children:X.name},X.path))]}),r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"4px",minWidth:"80px"},children:[r.jsx("input",{type:"range",min:"0",max:"2",step:"0.1",value:m.strength,onChange:X=>C(A,"strength",parseFloat(X.target.value)),disabled:m.name==="None",style:{width:"50px"}}),r.jsx("span",{style:{fontSize:"0.7rem",opacity:m.name==="None"?.3:1},children:m.strength.toFixed(1)})]})]},A)),r.jsx("div",{style:{fontSize:"0.65rem",opacity:.5,marginTop:"4px"},children:"Strength: 0.5-1.0 recommended"})]})]})]})]}),f&&r.jsx("div",{style:{color:"#ef4444",marginBottom:"12px",fontSize:"0.9rem"},children:f}),r.jsx("button",{className:"primary-btn",onClick:Y,disabled:z||!a.trim(),style:{height:"48px",fontSize:"1rem",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px",backgroundColor:"white",color:"black"},children:z?r.jsx(r.Fragment,{children:"Queueing..."}):r.jsxs(r.Fragment,{children:[r.jsx(Gt,{size:18}),"Generate ",k>1?`${k} Images`:"Image"," (",k,")"]})}),j&&r.jsxs("div",{style:{padding:"12px 16px",backgroundColor:"rgba(34, 197, 94, 0.2)",border:"1px solid rgba(34, 197, 94, 0.5)",borderRadius:"8px",color:"#86efac",fontSize:"0.875rem",marginTop:"12px"},children:["✅ ",j.count>1?`${j.count} jobs`:"Job"," queued! (",j.model,") - Check queue panel for progress"]}),f&&r.jsx("div",{style:{padding:"12px 16px",backgroundColor:"rgba(239, 68, 68, 0.2)",border:"1px solid rgba(239, 68, 68, 0.5)",borderRadius:"8px",color:"#fca5a5",fontSize:"0.875rem",marginTop:"12px"},children:f})]})}function og({onOutput:e}){const[t,n]=i.useState(""),[a,s]=i.useState("16:9"),[l,o]=i.useState(!1),[c,d]=i.useState(null),[p,v]=i.useState(""),[g,x]=i.useState(16),[k,w]=i.useState(!1),z=async()=>{t.trim()&&(o(!0),setTimeout(()=>{o(!1),alert("Text-to-Image backend is not yet connected.")},1500))},F=async()=>{c&&(w(!0),setTimeout(()=>w(!1),2e3))};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Step 1: Text to Image"}),r.jsx(gr,{size:16,className:"text-muted"})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Image Prompt"}),r.jsx("textarea",{className:"form-textarea",value:t,onChange:f=>n(f.target.value),placeholder:"Describe the image you want to generate...",rows:3,style:{backgroundColor:"#0f0f0f",border:"none",resize:"none"}})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Aspect Ratio"}),r.jsx("div",{className:"aspect-grid",children:[{label:"16:9",icon:r.jsx("div",{style:{width:"24px",height:"14px",border:"2px solid currentColor",borderRadius:"2px"}})},{label:"9:16",icon:r.jsx("div",{style:{width:"14px",height:"24px",border:"2px solid currentColor",borderRadius:"2px"}})},{label:"1:1",icon:r.jsx("div",{style:{width:"20px",height:"20px",border:"2px solid currentColor",borderRadius:"2px"}})},{label:"21:9",icon:r.jsx("div",{style:{width:"28px",height:"12px",border:"2px solid currentColor",borderRadius:"2px"}})}].map(f=>r.jsxs("button",{className:`aspect-btn ${a===f.label?"active":""}`,onClick:()=>s(f.label),children:[r.jsx("div",{className:"aspect-icon",style:{background:"transparent",display:"flex",alignItems:"center",justifyContent:"center"},children:f.icon}),r.jsx("span",{className:"aspect-label",children:f.label})]},f.label))})]}),r.jsx("button",{className:"primary-btn",onClick:z,disabled:l||!t.trim(),style:{display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:l?"Generating Image...":r.jsxs(r.Fragment,{children:[r.jsx(Gt,{size:16})," Generate Image"]})})]}),r.jsxs("div",{className:`grok-card ${c?"":"opacity-50"}`,style:{transition:"opacity 0.3s"},children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Step 2: Animate"}),r.jsx(da,{size:16,className:"text-muted"})]}),c?r.jsx("div",{className:"form-group",children:r.jsx("img",{src:c,alt:"Generated",style:{width:"100%",borderRadius:"8px",border:"1px solid var(--border-color)",marginBottom:"16px"}})}):r.jsx("div",{className:"upload-box",style:{padding:"24px",marginBottom:"16px",borderStyle:"dashed"},children:r.jsx("div",{className:"text-muted",children:"Generate an image above to continue"})}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Motion Prompt (Optional)"}),r.jsx("textarea",{className:"form-textarea",value:p,onChange:f=>v(f.target.value),placeholder:"Describe how the image should move...",rows:2,disabled:!c,style:{backgroundColor:"#0f0f0f",border:"none",resize:"none"}})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("label",{className:"grok-section-label",children:["Duration (",g," frames)"]}),r.jsx("input",{type:"range",min:"8",max:"32",step:"4",value:g,onChange:f=>x(parseInt(f.target.value,10)),disabled:!c,style:{width:"100%",accentColor:"var(--text-primary)"}})]}),r.jsx("button",{className:"primary-btn",onClick:F,disabled:!c||k,style:{display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:k?"Generating Video...":r.jsxs(r.Fragment,{children:[r.jsx(da,{size:16})," Generate Video"]})})]})]})}const ig=[{value:"none",label:"Custom",desc:"Use your own prompt"},{value:"anime",label:"Anime",desc:"Japanese animation style"},{value:"cartoon",label:"Cartoon",desc:"Cartoon/comic style"},{value:"sketch",label:"Sketch",desc:"Pencil sketch effect"},{value:"oil-painting",label:"Oil Painting",desc:"Classic oil painting style"},{value:"watercolor",label:"Watercolor",desc:"Watercolor painting effect"},{value:"pixel-art",label:"Pixel Art",desc:"Retro pixel art style"},{value:"cyberpunk",label:"Cyberpunk",desc:"Neon futuristic style"},{value:"3d-render",label:"3D Render",desc:"Modern 3D rendered look"}],cg={anime:"anime style, japanese animation, cel shading, vibrant colors, detailed linework",cartoon:"cartoon style, comic art, bold outlines, bright colors, disney style",sketch:"pencil sketch, hand-drawn, graphite, detailed linework, black and white","oil-painting":"oil painting style, classical art, brush strokes, rich colors, masterpiece",watercolor:"watercolor painting, soft edges, translucent colors, artistic, flowing","pixel-art":"pixel art style, 8-bit, retro gaming, blocky, nostalgic",cyberpunk:"cyberpunk style, neon lights, futuristic, rain, dark atmosphere, high tech","3d-render":"3d render, modern cgi, photorealistic, octane render, unreal engine"};function dg({onOutput:e,onJobSubmitted:t}){const[n,a]=i.useState(null),[s,l]=i.useState(null),[o,c]=i.useState(null),[d,p]=i.useState("none"),[v,g]=i.useState(""),[x,k]=i.useState("blurry, low quality, distorted, watermark"),[w,z]=i.useState(.5),[F,f]=i.useState(8),[u,h]=i.useState(32),[y,j]=i.useState(!1),[I,_]=i.useState(20),[R,G]=i.useState(7.5),[W,b]=i.useState(-1),[N,L]=i.useState(!1),[ee,T]=i.useState(null),[ne,ae]=i.useState(null),[D,U]=i.useState(null),q=i.useRef(null),V=i.useCallback(C=>{var M;const Y=(M=C.target.files)==null?void 0:M[0];if(Y){a(Y);const m=URL.createObjectURL(Y);l(m),U(null),T(null),ae(null);const A=document.createElement("video");A.onloadedmetadata=()=>{c({duration:A.duration.toFixed(1),width:A.videoWidth,height:A.videoHeight})},A.src=m}},[]),H=i.useCallback(C=>{var M;C.preventDefault();const Y=(M=C.dataTransfer.files)==null?void 0:M[0];if(Y&&Y.type.startsWith("video/")){a(Y);const m=URL.createObjectURL(Y);l(m),U(null),T(null),ae(null);const A=document.createElement("video");A.onloadedmetadata=()=>{c({duration:A.duration.toFixed(1),width:A.videoWidth,height:A.videoHeight})},A.src=m}},[]),Q=async()=>{var Y,M;if(!n)return;const C=d!=="none"?cg[d]+(v?", "+v:""):v;if(!C.trim()){T("Please select a style or enter a prompt");return}L(!0),T(null),ae(null);try{const m=new FormData;m.append("file",n),m.append("prompt",C),m.append("negative_prompt",x),m.append("denoise",String(w)),m.append("fps",String(F)),m.append("max_frames",String(u)),m.append("steps",String(I)),m.append("cfg",String(R)),m.append("seed",String(W));const A=await We(`${oe}/generate-v2v`,m);if(!A.ok)throw new Error(((Y=A.data)==null?void 0:Y.detail)||"V2V transform failed");const X=(M=A.data)==null?void 0:M.prompt_id;if(!X)throw new Error("No prompt_id returned");ae({promptId:X,style:d!=="none"?d:"custom"}),t&&t({prompt_id:X})}catch(m){console.error("V2V error:",m),T(m.message)}finally{L(!1)}};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(yr,{size:18}),"Source Video"]}),r.jsxs("div",{className:`upload-dropzone ${s?"has-preview":""}`,onDrop:H,onDragOver:C=>C.preventDefault(),onClick:()=>document.getElementById("v2v-file-input").click(),children:[s?r.jsx("video",{ref:q,src:s,className:"upload-preview",controls:!0,muted:!0,loop:!0,style:{maxHeight:"250px"}}):r.jsxs("div",{className:"upload-placeholder",children:[r.jsx(Ye,{size:32}),r.jsx("p",{children:"Drop video here or click to upload"}),r.jsx("span",{style:{fontSize:"12px",opacity:.6},children:"MP4, WebM, MOV"})]}),r.jsx("input",{id:"v2v-file-input",type:"file",accept:"video/*",onChange:V,style:{display:"none"}})]}),o&&r.jsxs("div",{className:"video-info",children:[r.jsxs("span",{children:["📐 ",o.width," × ",o.height,"px"]}),r.jsxs("span",{children:["⏱️ ",o.duration,"s"]})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(Kt,{size:18}),"Style Transform"]}),r.jsx("div",{className:"style-grid",children:ig.map(C=>r.jsxs("button",{className:`style-btn ${d===C.value?"active":""}`,onClick:()=>p(C.value),children:[r.jsx("span",{className:"style-name",children:C.label}),r.jsx("span",{className:"style-desc",children:C.desc})]},C.value))})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:["Prompt ",d!=="none"&&r.jsx("span",{className:"hint",children:"(optional - adds to style)"})]}),r.jsx("textarea",{value:v,onChange:C=>g(C.target.value),placeholder:d!=="none"?"Add extra details to the style...":"Describe the desired look...",rows:3,className:"prompt-textarea"})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Transform Strength"}),r.jsxs("div",{className:"slider-row",children:[r.jsx("input",{type:"range",min:"0.1",max:"1",step:"0.05",value:w,onChange:C=>z(parseFloat(C.target.value))}),r.jsxs("span",{className:"slider-value",children:[(w*100).toFixed(0),"%"]})]}),r.jsxs("div",{className:"slider-labels",children:[r.jsx("span",{children:"Subtle"}),r.jsx("span",{children:"Complete"})]})]}),r.jsxs("div",{className:"tool-section collapsible",children:[r.jsxs("h3",{onClick:()=>j(!y),style:{cursor:"pointer"},children:[r.jsx(vr,{size:16}),"Advanced Settings",r.jsx(Tt,{size:16,style:{marginLeft:"auto",transform:y?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.2s"}})]}),y&&r.jsxs("div",{className:"advanced-content",children:[r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Output FPS"}),r.jsxs("select",{value:F,onChange:C=>f(parseInt(C.target.value)),children:[r.jsx("option",{value:8,children:"8 fps"}),r.jsx("option",{value:12,children:"12 fps"}),r.jsx("option",{value:16,children:"16 fps"}),r.jsx("option",{value:24,children:"24 fps"})]})]}),r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Max Frames"}),r.jsxs("select",{value:u,onChange:C=>h(parseInt(C.target.value)),children:[r.jsx("option",{value:16,children:"16 frames (~2s @8fps)"}),r.jsx("option",{value:32,children:"32 frames (~4s @8fps)"}),r.jsx("option",{value:48,children:"48 frames (~6s @8fps)"}),r.jsx("option",{value:64,children:"64 frames (~8s @8fps)"})]})]}),r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Steps"}),r.jsx("input",{type:"number",min:10,max:50,value:I,onChange:C=>_(parseInt(C.target.value))})]}),r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"CFG Scale"}),r.jsx("input",{type:"number",min:1,max:15,step:.5,value:R,onChange:C=>G(parseFloat(C.target.value))})]}),r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Seed (-1 = random)"}),r.jsx("input",{type:"number",value:W,onChange:C=>b(parseInt(C.target.value)||-1)})]}),r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Negative Prompt"}),r.jsx("textarea",{value:x,onChange:C=>k(C.target.value),rows:2,style:{fontSize:"12px"}})]})]})]}),ne&&r.jsxs("div",{className:"queued-notice",children:["✅ Job queued! Check the Queue panel for progress.",r.jsx("span",{className:"queued-mode",children:ne.style.toUpperCase()})]}),ee&&r.jsxs("div",{className:"error-message",children:["⚠️ ",ee]}),r.jsx("button",{className:"btn-primary btn-large",onClick:Q,disabled:!n||N,children:N?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Queueing..."]}):r.jsxs(r.Fragment,{children:[r.jsx(Kt,{size:18}),"Transform Video"]})}),D&&r.jsxs("div",{className:"result-section",children:[r.jsx("h3",{children:"Result"}),r.jsx("video",{src:D,controls:!0,className:"result-video"}),r.jsx("a",{href:D,download:!0,className:"btn-secondary",style:{marginTop:12},children:"Download Video"})]}),r.jsx("style",{children:`
        .tool-section {
          margin-bottom: 20px;
        }
        .tool-section h3 {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          font-weight: 500;
          margin-bottom: 12px;
          color: var(--text-color, #fff);
        }
        .tool-section h3 .hint {
          font-weight: 400;
          font-size: 12px;
          color: var(--text-muted, #666);
        }
        .upload-dropzone {
          border: 2px dashed var(--border-color, #444);
          border-radius: 12px;
          padding: 40px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
          min-height: 150px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .upload-dropzone:hover {
          border-color: var(--accent-color, #7c3aed);
          background: rgba(124, 58, 237, 0.05);
        }
        .upload-dropzone.has-preview {
          padding: 8px;
        }
        .upload-preview {
          max-width: 100%;
          border-radius: 8px;
        }
        .upload-placeholder {
          color: var(--text-muted, #888);
        }
        .upload-placeholder p {
          margin-top: 12px;
          margin-bottom: 4px;
        }
        .video-info {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 16px;
          margin-top: 12px;
          font-size: 13px;
          color: var(--text-muted, #888);
        }
        .style-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }
        .style-btn {
          padding: 12px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: var(--bg-secondary, #1a1a1a);
          cursor: pointer;
          transition: all 0.2s;
          text-align: left;
        }
        .style-btn:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .style-btn.active {
          background: rgba(124, 58, 237, 0.2);
          border-color: var(--accent-color, #7c3aed);
        }
        .style-name {
          display: block;
          font-size: 13px;
          font-weight: 500;
          color: var(--text-color, #fff);
        }
        .style-desc {
          display: block;
          font-size: 11px;
          color: var(--text-muted, #888);
          margin-top: 4px;
        }
        .prompt-textarea {
          width: 100%;
          padding: 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 13px;
          resize: none;
        }
        .slider-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .slider-row input[type="range"] {
          flex: 1;
        }
        .slider-value {
          min-width: 45px;
          text-align: right;
          font-weight: 500;
          color: var(--accent-color, #7c3aed);
        }
        .slider-labels {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: var(--text-muted, #666);
          margin-top: 4px;
        }
        .collapsible h3 {
          padding: 12px;
          margin: -12px -12px 0;
          border-radius: 8px;
        }
        .collapsible h3:hover {
          background: var(--bg-secondary, #1a1a1a);
        }
        .advanced-content {
          margin-top: 12px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .form-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .form-row label {
          min-width: 120px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .form-row select, .form-row input {
          flex: 1;
          padding: 8px 12px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 13px;
        }
        .form-row textarea {
          flex: 1;
          padding: 8px 12px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          resize: none;
        }
        .progress-section {
          margin: 16px 0;
        }
        .progress-bar {
          height: 4px;
          background: var(--bg-secondary, #333);
          border-radius: 2px;
          overflow: hidden;
        }
        .progress-fill {
          height: 100%;
          background: var(--accent-color, #7c3aed);
          transition: width 0.3s;
        }
        .progress-status {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .error-message {
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          margin: 12px 0;
        }
        .result-section {
          margin-top: 24px;
          padding-top: 24px;
          border-top: 1px solid var(--border-color, #333);
        }
        .result-video {
          width: 100%;
          border-radius: 8px;
          margin-top: 12px;
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @media (max-width: 600px) {
          .style-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }
      `})]})}const ug=[{value:"brief",label:"Brief",desc:"Short 1-2 sentence description"},{value:"detailed",label:"Detailed",desc:"Comprehensive scene analysis"},{value:"prompt",label:"Prompt Style",desc:"Optimized for AI generation"},{value:"timeline",label:"Timeline",desc:"Frame-by-frame breakdown"}],pg=[{value:"smolvlm",label:"SmolVLM",desc:"Fast, lightweight vision model"},{value:"cogvlm",label:"CogVLM",desc:"High quality, slower"},{value:"llava",label:"LLaVA",desc:"Balanced quality/speed"}],fg=[{value:"upload",label:"Upload",icon:Ye},{value:"youtube",label:"YouTube",icon:Ox}];function mg(){var X;const[e,t]=i.useState("upload"),[n,a]=i.useState(null),[s,l]=i.useState(null),[o,c]=i.useState(null),[d,p]=i.useState(""),[v,g]=i.useState(null),[x,k]=i.useState(!1),[w,z]=i.useState(null),[F,f]=i.useState("smolvlm"),[u,h]=i.useState("detailed"),[y,j]=i.useState(1),[I,_]=i.useState(8),[R,G]=i.useState(!1),[W,b]=i.useState(!1),[N,L]=i.useState(null),[ee,T]=i.useState(""),[ne,ae]=i.useState(null),[D,U]=i.useState(!1),q=i.useRef(null),V=i.useCallback(P=>{var te;const O=(te=P.target.files)==null?void 0:te[0];if(O){a(O);const K=URL.createObjectURL(O);l(K),ae(null),L(null);const de=document.createElement("video");de.onloadedmetadata=()=>{c({duration:de.duration.toFixed(1),width:de.videoWidth,height:de.videoHeight})},de.src=K}},[]),H=i.useCallback(P=>{var te;P.preventDefault();const O=(te=P.dataTransfer.files)==null?void 0:te[0];if(O&&O.type.startsWith("video/")){a(O);const K=URL.createObjectURL(O);l(K),ae(null),L(null),z(null);const de=document.createElement("video");de.onloadedmetadata=()=>{c({duration:de.duration.toFixed(1),width:de.videoWidth,height:de.videoHeight})},de.src=K}},[]),Q=P=>/^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+/.test(P),C=P=>{const O=P.target.value;p(O),g(null),L(null)},Y=async()=>{var P;if(!d||!Q(d)){L("Please enter a valid YouTube URL");return}k(!0),L(null);try{const O=await fa(`${oe}/youtube/info`,{url:d});if(!O.ok)throw new Error(((P=O.data)==null?void 0:P.detail)||"Failed to fetch video info");g(O.data)}catch(O){L(O.message)}finally{k(!1)}},M=async()=>{var P,O;if(d){k(!0),L(null),T("Downloading video from YouTube...");try{const te=await fa(`${oe}/youtube/download`,{url:d,format:"video",quality:"720p"});if(!te.ok)throw new Error(((P=te.data)==null?void 0:P.detail)||"Failed to download video");z(te.data.path),l(`${oe}/file/${encodeURIComponent(te.data.path)}`),c({duration:((O=te.data.duration)==null?void 0:O.toFixed(1))||(v==null?void 0:v.duration),width:te.data.width||(v==null?void 0:v.width)||1280,height:te.data.height||(v==null?void 0:v.height)||720,title:v==null?void 0:v.title})}catch(te){L(te.message)}finally{k(!1),T("")}}},m=async()=>{var P;if(!(!n&&!w)){b(!0),L(null),T("Uploading video...");try{const O=new FormData;w?O.append("video_path",w):O.append("file",n),O.append("model",F),O.append("mode",u),O.append("frame_interval",String(y)),O.append("max_frames",String(I)),T("Analyzing video...");const te=await We(`${oe}/caption-video`,O);if(!te.ok)throw new Error(((P=te.data)==null?void 0:P.detail)||"Video analysis failed");ae(te.data)}catch(O){console.error("V2T error:",O),L(O.message)}finally{b(!1),T("")}}},A=async P=>{await navigator.clipboard.writeText(P),U(!0),setTimeout(()=>U(!1),2e3)};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(yr,{size:18}),"Source Video"]}),r.jsx("div",{className:"source-tabs",children:fg.map(P=>r.jsxs("button",{className:`source-tab ${e===P.value?"active":""}`,onClick:()=>{t(P.value),L(null)},children:[r.jsx(P.icon,{size:16}),P.label]},P.value))}),e==="upload"&&r.jsxs("div",{className:`upload-dropzone ${s?"has-preview":""}`,onDrop:H,onDragOver:P=>P.preventDefault(),onClick:()=>document.getElementById("v2t-file-input").click(),children:[s&&!w?r.jsx("video",{ref:q,src:s,className:"upload-preview",controls:!0,muted:!0,style:{maxHeight:"200px"}}):r.jsxs("div",{className:"upload-placeholder",children:[r.jsx(Ye,{size:32}),r.jsx("p",{children:"Drop video here or click to upload"}),r.jsx("span",{style:{fontSize:"12px",opacity:.6},children:"MP4, WebM, MOV"})]}),r.jsx("input",{id:"v2t-file-input",type:"file",accept:"video/*",onChange:V,style:{display:"none"}})]}),e==="youtube"&&r.jsxs("div",{className:"youtube-section",children:[r.jsxs("div",{className:"youtube-input-row",children:[r.jsxs("div",{className:"youtube-input-wrapper",children:[r.jsx(dp,{size:16,className:"youtube-input-icon"}),r.jsx("input",{type:"text",className:"youtube-input",placeholder:"Paste YouTube URL here...",value:d,onChange:C,onKeyDown:P=>P.key==="Enter"&&Y()})]}),r.jsx("button",{className:"btn-secondary",onClick:Y,disabled:x||!d,children:x?r.jsx(Oe,{size:16,className:"spin"}):"Fetch"})]}),v&&r.jsxs("div",{className:"youtube-preview",children:[v.thumbnail&&r.jsx("img",{src:v.thumbnail,alt:"thumbnail",className:"youtube-thumbnail"}),r.jsxs("div",{className:"youtube-info",children:[r.jsx("span",{className:"youtube-title",children:v.title}),r.jsxs("span",{className:"youtube-meta",children:[v.channel," • ",v.duration,"s • ",(X=v.view_count)==null?void 0:X.toLocaleString()," views"]})]}),r.jsx("button",{className:"btn-primary",onClick:M,disabled:x,children:x?r.jsx(Oe,{size:16,className:"spin"}):r.jsxs(r.Fragment,{children:[r.jsx(vt,{size:16}),"Download"]})})]}),w&&r.jsxs("div",{className:"youtube-downloaded",children:[r.jsx(Ns,{size:16,style:{color:"#22c55e"}}),r.jsx("span",{children:"Video ready for analysis"}),s&&r.jsx("video",{src:s,className:"upload-preview",controls:!0,muted:!0,style:{maxHeight:"200px",marginTop:"12px",width:"100%"}})]})]}),o&&r.jsxs("div",{className:"video-info",children:[r.jsxs("span",{children:["📐 ",o.width," × ",o.height]}),r.jsxs("span",{children:["⏱️ ",o.duration,"s"]})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(Rc,{size:18}),"Analysis Model"]}),r.jsx("div",{className:"model-grid",children:pg.map(P=>r.jsxs("button",{className:`model-btn ${F===P.value?"active":""}`,onClick:()=>f(P.value),children:[r.jsx("span",{className:"model-name",children:P.label}),r.jsx("span",{className:"model-desc",children:P.desc})]},P.value))})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Output Style"}),r.jsx("div",{className:"mode-grid",children:ug.map(P=>r.jsxs("button",{className:`mode-btn ${u===P.value?"active":""}`,onClick:()=>h(P.value),children:[r.jsx("span",{className:"mode-name",children:P.label}),r.jsx("span",{className:"mode-desc",children:P.desc})]},P.value))})]}),r.jsxs("div",{className:"tool-section collapsible",children:[r.jsxs("h3",{onClick:()=>G(!R),style:{cursor:"pointer"},children:[r.jsx(vr,{size:16}),"Advanced",r.jsx(Tt,{size:16,style:{marginLeft:"auto",transform:R?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.2s"}})]}),R&&r.jsxs("div",{className:"advanced-content",children:[r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Frame Interval"}),r.jsxs("select",{value:y,onChange:P=>j(parseFloat(P.target.value)),children:[r.jsx("option",{value:.5,children:"Every 0.5s"}),r.jsx("option",{value:1,children:"Every 1s"}),r.jsx("option",{value:2,children:"Every 2s"}),r.jsx("option",{value:5,children:"Every 5s"})]})]}),r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Max Frames"}),r.jsxs("select",{value:I,onChange:P=>_(parseInt(P.target.value)),children:[r.jsx("option",{value:4,children:"4 frames"}),r.jsx("option",{value:8,children:"8 frames"}),r.jsx("option",{value:16,children:"16 frames"}),r.jsx("option",{value:32,children:"32 frames"})]})]})]})]}),N&&r.jsxs("div",{className:"error-message",children:["⚠️ ",N]}),r.jsx("button",{className:"btn-primary btn-large",onClick:m,disabled:!n&&!w||W,children:W?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),ee]}):r.jsxs(r.Fragment,{children:[r.jsx(Rc,{size:18}),"Analyze Video"]})}),ne&&r.jsxs("div",{className:"result-section",children:[r.jsxs("div",{className:"result-header",children:[r.jsx("h3",{children:"Description"}),r.jsxs("button",{className:"copy-btn",onClick:()=>A(ne.caption||ne.description),children:[D?r.jsx(Ns,{size:16}):r.jsx(Wt,{size:16}),D?"Copied!":"Copy"]})]}),r.jsx("div",{className:"result-text",children:ne.caption||ne.description}),ne.timeline&&ne.timeline.length>0&&r.jsxs("div",{className:"timeline-section",children:[r.jsx("h4",{children:"Timeline"}),ne.timeline.map((P,O)=>r.jsxs("div",{className:"timeline-item",children:[r.jsxs("span",{className:"timeline-time",children:[P.time,"s"]}),r.jsx("span",{className:"timeline-desc",children:P.description})]},O))]}),ne.prompt&&r.jsxs("div",{className:"prompt-section",children:[r.jsxs("div",{className:"prompt-header",children:[r.jsx("h4",{children:"AI Generation Prompt"}),r.jsx("button",{className:"copy-btn small",onClick:()=>A(ne.prompt),children:r.jsx(Wt,{size:14})})]}),r.jsx("div",{className:"prompt-text",children:ne.prompt})]})]}),r.jsx("style",{children:`
        .tool-section {
          margin-bottom: 20px;
        }
        .tool-section h3 {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          font-weight: 500;
          margin-bottom: 12px;
          color: var(--text-color, #fff);
        }
        .source-tabs {
          display: flex;
          gap: 8px;
          margin-bottom: 12px;
        }
        .source-tab {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 16px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: transparent;
          color: var(--text-muted, #888);
          cursor: pointer;
          transition: all 0.2s;
          font-size: 13px;
        }
        .source-tab:hover {
          border-color: var(--accent-color, #7c3aed);
          color: var(--text-color, #fff);
        }
        .source-tab.active {
          background: rgba(124, 58, 237, 0.2);
          border-color: var(--accent-color, #7c3aed);
          color: var(--text-color, #fff);
        }
        .youtube-section {
          padding: 16px;
          border: 1px solid var(--border-color, #444);
          border-radius: 12px;
          background: var(--bg-secondary, #1a1a1a);
        }
        .youtube-input-row {
          display: flex;
          gap: 8px;
        }
        .youtube-input-wrapper {
          flex: 1;
          position: relative;
        }
        .youtube-input-icon {
          position: absolute;
          left: 12px;
          top: 50%;
          transform: translateY(-50%);
          color: var(--text-muted, #888);
        }
        .youtube-input {
          width: 100%;
          padding: 10px 12px 10px 36px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: var(--bg-primary, #0a0a0a);
          color: var(--text-color, #fff);
          font-size: 13px;
        }
        .youtube-input:focus {
          outline: none;
          border-color: var(--accent-color, #7c3aed);
        }
        .youtube-preview {
          display: flex;
          align-items: center;
          gap: 12px;
          margin-top: 12px;
          padding: 12px;
          background: var(--bg-primary, #0a0a0a);
          border-radius: 8px;
        }
        .youtube-thumbnail {
          width: 120px;
          height: 68px;
          object-fit: cover;
          border-radius: 6px;
        }
        .youtube-info {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .youtube-title {
          font-size: 13px;
          font-weight: 500;
          color: var(--text-color, #fff);
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
        .youtube-meta {
          font-size: 11px;
          color: var(--text-muted, #888);
        }
        .youtube-downloaded {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 12px;
          padding: 12px;
          background: rgba(34, 197, 94, 0.1);
          border: 1px solid rgba(34, 197, 94, 0.3);
          border-radius: 8px;
          color: #22c55e;
          font-size: 13px;
          flex-wrap: wrap;
        }
        .upload-dropzone {
          border: 2px dashed var(--border-color, #444);
          border-radius: 12px;
          padding: 40px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
          min-height: 120px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .upload-dropzone:hover {
          border-color: var(--accent-color, #7c3aed);
          background: rgba(124, 58, 237, 0.05);
        }
        .upload-dropzone.has-preview {
          padding: 8px;
        }
        .upload-preview {
          max-width: 100%;
          border-radius: 8px;
        }
        .upload-placeholder {
          color: var(--text-muted, #888);
        }
        .video-info {
          display: flex;
          gap: 16px;
          justify-content: center;
          margin-top: 8px;
          font-size: 12px;
          color: var(--text-muted, #888);
        }
        .model-grid, .mode-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }
        .mode-grid {
          grid-template-columns: repeat(2, 1fr);
        }
        .model-btn, .mode-btn {
          padding: 12px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: var(--bg-secondary, #1a1a1a);
          cursor: pointer;
          transition: all 0.2s;
          text-align: left;
        }
        .model-btn:hover, .mode-btn:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .model-btn.active, .mode-btn.active {
          background: rgba(124, 58, 237, 0.2);
          border-color: var(--accent-color, #7c3aed);
        }
        .model-name, .mode-name {
          display: block;
          font-size: 13px;
          font-weight: 500;
          color: var(--text-color, #fff);
        }
        .model-desc, .mode-desc {
          display: block;
          font-size: 11px;
          color: var(--text-muted, #888);
          margin-top: 2px;
        }
        .collapsible h3 {
          padding: 12px;
          margin: -12px -12px 0;
          border-radius: 8px;
        }
        .collapsible h3:hover {
          background: var(--bg-secondary, #1a1a1a);
        }
        .advanced-content {
          margin-top: 12px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .form-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .form-row label {
          min-width: 100px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .form-row select {
          flex: 1;
          padding: 8px;
          border-radius: 6px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
        }
        .error-message {
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          margin: 12px 0;
        }
        .result-section {
          margin-top: 24px;
          padding-top: 24px;
          border-top: 1px solid var(--border-color, #333);
        }
        .result-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .result-header h3 {
          margin: 0;
        }
        .copy-btn {
          display: flex;
          align-items: center;
          gap: 4px;
          padding: 6px 12px;
          border: 1px solid var(--border-color, #444);
          border-radius: 6px;
          background: transparent;
          color: var(--text-color, #fff);
          cursor: pointer;
          font-size: 12px;
        }
        .copy-btn:hover {
          background: var(--bg-secondary, #1a1a1a);
        }
        .copy-btn.small {
          padding: 4px 8px;
        }
        .result-text {
          padding: 16px;
          background: var(--bg-secondary, #1a1a1a);
          border-radius: 8px;
          font-size: 14px;
          line-height: 1.6;
          white-space: pre-wrap;
        }
        .timeline-section {
          margin-top: 16px;
        }
        .timeline-section h4 {
          font-size: 13px;
          margin-bottom: 8px;
          color: var(--text-secondary, #aaa);
        }
        .timeline-item {
          display: flex;
          gap: 12px;
          padding: 8px 0;
          border-bottom: 1px solid var(--border-color, #333);
        }
        .timeline-time {
          min-width: 50px;
          font-weight: 500;
          color: var(--accent-color, #7c3aed);
        }
        .timeline-desc {
          color: var(--text-color, #fff);
          font-size: 13px;
        }
        .prompt-section {
          margin-top: 16px;
          padding: 12px;
          background: rgba(124, 58, 237, 0.1);
          border: 1px solid rgba(124, 58, 237, 0.3);
          border-radius: 8px;
        }
        .prompt-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }
        .prompt-header h4 {
          margin: 0;
          font-size: 12px;
          color: var(--accent-color, #7c3aed);
        }
        .prompt-text {
          font-size: 13px;
          color: var(--text-color, #fff);
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `})]})}const hg=["video/mp4","video/webm","video/quicktime"],xg=[{id:"f5v1",label:"F5-TTS v1",description:"Fast, high quality"},{id:"e2",label:"E2-TTS",description:"More expressive"}],gg=[{id:"custom",label:"Upload Voice Sample",isCustom:!0},{id:"alloy",label:"Alloy (Neutral)"},{id:"echo",label:"Echo (Male)"},{id:"fable",label:"Fable (British)"},{id:"onyx",label:"Onyx (Deep Male)"},{id:"nova",label:"Nova (Female)"},{id:"shimmer",label:"Shimmer (Soft Female)"}];function vg({onOutput:e,onJobSubmitted:t}){const[n,a]=i.useState(null),[s,l]=i.useState(null),[o,c]=i.useState(null),[d,p]=i.useState(""),[v,g]=i.useState("f5v1"),[x,k]=i.useState("nova"),[w,z]=i.useState(null),[F,f]=i.useState(null),[u,h]=i.useState(1.5),[y,j]=i.useState(20),[I,_]=i.useState(!1),[R,G]=i.useState(!1),[W,b]=i.useState(!1),[N,L]=i.useState(null),[ee,T]=i.useState(null),[ne,ae]=i.useState(null),D=i.useRef(null),U=i.useRef(null),q=i.useRef(null),V=i.useCallback(m=>{var X,P,O,te;m.preventDefault();const A=((P=(X=m.dataTransfer)==null?void 0:X.files)==null?void 0:P[0])||((te=(O=m.target)==null?void 0:O.files)==null?void 0:te[0]);A&&hg.some(K=>A.type.includes(K.split("/")[1]))?(a(A),l(URL.createObjectURL(A)),c(null),T(null),ae(null)):A&&T("Please upload a valid video file (MP4, WebM)")},[]),H=i.useCallback(m=>{var X,P,O,te;m.preventDefault();const A=((P=(X=m.dataTransfer)==null?void 0:X.files)==null?void 0:P[0])||((te=(O=m.target)==null?void 0:O.files)==null?void 0:te[0]);A&&A.type.startsWith("audio/")?(z(A),f(URL.createObjectURL(A)),T(null)):A&&T("Please upload a valid audio file for voice sample")},[]),Q=async m=>{var X,P;const A=new FormData;A.append("file",m);try{const O=await We(`${oe}/upload`,A);if(O.ok&&((X=O.data)!=null&&X.path))return O.data.path;throw new Error(((P=O.data)==null?void 0:P.detail)||"Upload failed")}catch(O){throw new Error(`Upload failed: ${O.message}`)}},C=async()=>{var m,A,X,P,O,te;if(!n||!d.trim()){T("Please upload a video and enter text");return}G(!0),T(null),ae(null);try{L("Uploading video..."),b(!0);let K=o;K||(K=await Q(n),c(K));let de=null;x==="custom"&&w&&(L("Uploading voice sample..."),de=await Q(w)),b(!1),L("Generating speech...");const pe=new FormData;pe.append("text",d),pe.append("model",v),x==="custom"&&de?pe.append("voice_sample",de):x!=="custom"&&pe.append("voice_preset",x);const Te=await We(`${oe}/voice-clone`,pe);if(!Te.ok)throw new Error(((m=Te.data)==null?void 0:m.detail)||"TTS generation failed");const nt=((A=Te.data)==null?void 0:A.path)||((X=Te.data)==null?void 0:X.audio_path);if(!nt)throw new Error("TTS did not return audio path");L("Applying lip sync...");const Pt={video_path:K,audio_path:nt,lips_expression:u,inference_steps:y,seed:-1},bt=await fa(`${oe}/lip-sync`,Pt);if(!bt.ok)throw new Error(((P=bt.data)==null?void 0:P.detail)||"Lip sync failed");ae({promptId:(O=bt.data)==null?void 0:O.prompt_id,text:d.slice(0,30)+(d.length>30?"...":"")}),t&&t({prompt_id:(te=bt.data)==null?void 0:te.prompt_id})}catch(K){console.error("❌ Speech-to-Video error:",K),T(K.message)}finally{G(!1),b(!1),L(null)}},Y=()=>{a(null),l(null),c(null),ae(null)},M=()=>{z(null),f(null)};return r.jsxs("div",{className:"tool-container space-y-4 p-4",children:[r.jsxs("div",{className:"text-center mb-4",children:[r.jsxs("h2",{className:"text-xl font-bold text-white flex items-center justify-center gap-2",children:[r.jsx(jl,{className:"w-6 h-6 text-purple-400"}),"Speech to Video"]}),r.jsx("p",{className:"text-gray-400 text-sm mt-1",children:"Generate speech from text and sync it to a video"})]}),r.jsxs("div",{onClick:()=>{var m;return(m=U.current)==null?void 0:m.click()},onDrop:V,onDragOver:m=>m.preventDefault(),className:"border-2 border-dashed border-gray-600 rounded-lg p-6 text-center cursor-pointer hover:border-purple-500 transition-colors",children:[r.jsx("input",{ref:U,type:"file",accept:"video/*",onChange:V,className:"hidden"}),s?r.jsxs("div",{className:"space-y-2",children:[r.jsx("video",{ref:D,src:s,className:"max-h-40 mx-auto rounded",controls:!0,muted:!0}),r.jsxs("div",{className:"flex items-center justify-center gap-2",children:[r.jsx("span",{className:"text-sm text-gray-400",children:n==null?void 0:n.name}),r.jsx("button",{onClick:m=>{m.stopPropagation(),Y()},className:"p-1 text-red-400 hover:text-red-300",children:r.jsx(Qe,{className:"w-4 h-4"})})]})]}):r.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[r.jsx(yr,{className:"w-10 h-10"}),r.jsx("span",{children:"Drop video here or click to upload"}),r.jsx("span",{className:"text-xs",children:"MP4, WebM supported"})]})]}),r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:[r.jsx(jl,{className:"w-4 h-4 inline mr-1"}),"Text to Speak"]}),r.jsx("textarea",{value:d,onChange:m=>p(m.target.value),placeholder:"Enter the text you want the character to say...",className:"w-full px-3 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 resize-none",rows:4}),r.jsxs("div",{className:"text-xs text-gray-500 mt-1 text-right",children:[d.length," characters"]})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"TTS Model"}),r.jsx("div",{className:"grid grid-cols-2 gap-2",children:xg.map(m=>r.jsxs("button",{onClick:()=>g(m.id),className:`px-3 py-2 text-sm rounded transition-colors text-left ${v===m.id?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:[r.jsx("div",{className:"font-medium",children:m.label}),r.jsx("div",{className:"text-xs opacity-70",children:m.description})]},m.id))})]}),r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:[r.jsx(gi,{className:"w-4 h-4 inline mr-1"}),"Voice"]}),r.jsx("select",{value:x,onChange:m=>k(m.target.value),className:"w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white",children:gg.map(m=>r.jsx("option",{value:m.id,children:m.label},m.id))})]}),x==="custom"&&r.jsxs("div",{onClick:()=>{var m;return(m=q.current)==null?void 0:m.click()},onDrop:H,onDragOver:m=>m.preventDefault(),className:"border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-purple-500 transition-colors",children:[r.jsx("input",{ref:q,type:"file",accept:"audio/*",onChange:H,className:"hidden"}),F?r.jsxs("div",{className:"space-y-2",children:[r.jsx("audio",{src:F,controls:!0,className:"mx-auto"}),r.jsxs("div",{className:"flex items-center justify-center gap-2",children:[r.jsx("span",{className:"text-sm text-gray-400",children:w==null?void 0:w.name}),r.jsx("button",{onClick:m=>{m.stopPropagation(),M()},className:"p-1 text-red-400 hover:text-red-300",children:r.jsx(Qe,{className:"w-4 h-4"})})]})]}):r.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[r.jsx(hn,{className:"w-6 h-6"}),r.jsx("span",{className:"text-sm",children:"Upload voice sample (5-15 sec recommended)"})]})]}),r.jsxs("div",{className:"border border-gray-700 rounded-lg overflow-hidden",children:[r.jsxs("button",{onClick:()=>_(!I),className:"w-full px-4 py-2 bg-gray-800 flex items-center justify-between text-gray-300 hover:bg-gray-750",children:[r.jsxs("span",{className:"text-sm font-medium flex items-center gap-2",children:[r.jsx(fp,{className:"w-4 h-4"}),"Lip Sync Settings"]}),r.jsx(Tt,{className:`w-4 h-4 transition-transform ${I?"rotate-180":""}`})]}),I&&r.jsxs("div",{className:"p-4 space-y-4 bg-gray-850",children:[r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Lips Expression: ",u.toFixed(1)]}),r.jsx("input",{type:"range",min:.5,max:3,step:.1,value:u,onChange:m=>h(parseFloat(m.target.value)),className:"w-full accent-purple-500"}),r.jsx("span",{className:"text-xs text-gray-500",children:"Higher = more pronounced lip movement"})]}),r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Inference Steps: ",y]}),r.jsx("input",{type:"range",min:10,max:50,step:5,value:y,onChange:m=>j(parseInt(m.target.value)),className:"w-full accent-purple-500"}),r.jsx("span",{className:"text-xs text-gray-500",children:"Higher = better quality, slower"})]})]})]}),r.jsx("button",{onClick:C,disabled:R||!n||!d.trim(),className:"w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors",children:R?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{className:"w-5 h-5 animate-spin"}),N||"Processing..."]}):r.jsxs(r.Fragment,{children:[r.jsx(jl,{className:"w-5 h-5"}),"Generate Speech Video"]})}),ne&&r.jsxs("div",{className:"p-3 bg-green-900/50 border border-green-700 rounded-lg text-green-200 text-sm",children:['✅ Speech-to-Video queued! "',ne.text,'" - Check queue panel for progress']}),ee&&r.jsx("div",{className:"p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm",children:ee}),r.jsx("div",{className:"text-xs text-gray-500 text-center",children:"This tool generates speech from your text using TTS, then applies lip sync to match the video."})]})}function yg(){var s;const[e,t]=i.useState([{id:1,name:"Text Generation",status:"completed",description:"Generate prompt from keywords"},{id:2,name:"Text to Image",status:"ready",description:"Create base image"},{id:3,name:"Image to Video",status:"pending",description:"Animate the image"},{id:4,name:"Upscale",status:"pending",description:"Enhance resolution"}]),[n,a]=i.useState(2);return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Production Pipeline"}),r.jsx(hp,{size:16,className:"text-muted"})]}),r.jsx("div",{style:{display:"flex",flexDirection:"column",gap:"16px"},children:e.map((l,o)=>r.jsxs("div",{className:`pipeline-step ${n===l.id?"active":""}`,style:{display:"flex",alignItems:"center",gap:"16px",padding:"16px",backgroundColor:n===l.id?"#1a1a1a":"transparent",borderRadius:"8px",border:n===l.id?"1px solid var(--border-color)":"1px solid transparent",opacity:l.status==="pending"?.5:1},children:[r.jsx("div",{style:{width:"32px",height:"32px",borderRadius:"50%",backgroundColor:l.status==="completed"?"#22c55e":n===l.id?"var(--text-primary)":"#333",color:l.status==="completed"||n===l.id?"var(--bg-root)":"var(--text-secondary)",display:"flex",alignItems:"center",justifyContent:"center",fontWeight:"bold",fontSize:"0.9rem"},children:l.status==="completed"?r.jsx(vh,{size:18}):l.id}),r.jsxs("div",{style:{flex:1},children:[r.jsx("div",{style:{fontWeight:600,color:"var(--text-primary)"},children:l.name}),r.jsx("div",{style:{fontSize:"0.85rem",color:"var(--text-secondary)"},children:l.description})]}),o<e.length-1&&r.jsx(ah,{size:16,className:"text-muted",style:{opacity:.3}})]},l.id))})]}),r.jsxs("div",{className:"grok-card",children:[r.jsx("div",{className:"grok-card-header",children:r.jsxs("div",{className:"grok-card-title",children:["Step Configuration: ",(s=e.find(l=>l.id===n))==null?void 0:s.name]})}),r.jsx("div",{className:"placeholder-state",style:{padding:"20px 0"},children:r.jsx("div",{className:"text-muted",children:"Configuration options for this step would appear here."})})]}),r.jsxs("button",{className:"primary-btn",style:{display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:[r.jsx(ua,{size:18}),"Run Pipeline"]})]})}function jg({onOutput:e}){const t=i.useRef(null),[n,a]=i.useState([]),[s,l]=i.useState(""),[o,c]=i.useState(10),[d,p]=i.useState(1e-4),[v,g]=i.useState(!1),[x,k]=i.useState(""),w=i.useMemo(()=>n.length>0&&s.trim().length>0&&!v,[n,s,v]),z=u=>{const h=Array.from(u||[]);a(h),k("")},F=()=>{a([]),t.current&&(t.current.value="")},f=async()=>{var h;if(n.length===0){k("At least one image is required");return}if(!s.trim()){k("Model name is required");return}g(!0),k("");const u=new FormData;n.forEach(y=>u.append("files",y)),u.append("model_name",s.trim()),u.append("num_epochs",String(o)),u.append("learning_rate",String(d));try{const y=await We(`${oe}/train-lora`,u);if(!y.ok){k(((h=y.data)==null?void 0:h.detail)||`Training failed (status ${y.status})`);return}e({kind:"lora",...y.data})}catch(y){const j=(y==null?void 0:y.message)||"Failed to start LoRA training";k(j),await Ws({level:"error",message:"LoRA training failed",timestamp:new Date().toISOString(),meta:{message:j}})}finally{g(!1)}};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Training Dataset"}),r.jsx(xi,{size:16,className:"text-muted"})]}),r.jsx("input",{ref:t,type:"file",accept:"image/*",multiple:!0,onChange:u=>z(u.target.files),style:{display:"none"}}),n.length===0?r.jsxs("div",{className:"upload-box",onClick:()=>{var u;return(u=t.current)==null?void 0:u.click()},style:{cursor:"pointer"},children:[r.jsx(Ye,{size:32,className:"text-muted"}),r.jsx("div",{className:"text-muted",children:"Upload training images (5-20 recommended)"}),r.jsxs("button",{className:"upload-btn",children:[r.jsx(Ye,{size:16}),"Select Images"]})]}):r.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"12px"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsxs("span",{style:{color:"var(--text-primary)",fontWeight:500},children:[n.length," images selected"]}),r.jsxs("button",{onClick:F,className:"upload-btn secondary",style:{padding:"4px 8px",fontSize:"0.8rem"},children:[r.jsx(Qe,{size:14})," Clear"]})]}),r.jsx("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fill, minmax(60px, 1fr))",gap:"8px",maxHeight:"200px",overflowY:"auto",padding:"8px",backgroundColor:"#0f0f0f",borderRadius:"8px",border:"1px solid var(--border-color)"},children:n.map((u,h)=>r.jsx("div",{style:{aspectRatio:"1/1",backgroundColor:"#222",borderRadius:"4px",overflow:"hidden",display:"flex",alignItems:"center",justifyContent:"center"},children:r.jsx("span",{style:{fontSize:"0.6rem",color:"#666"},children:"IMG"})},h))})]})]}),r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Configuration"}),r.jsx(vr,{size:16,className:"text-muted"})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Model Name"}),r.jsx("input",{className:"form-input",value:s,onChange:u=>l(u.target.value),placeholder:"e.g. my-style-v1",style:{backgroundColor:"#0f0f0f"}})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("label",{className:"grok-section-label",children:["Training Epochs (",o,")"]}),r.jsx("input",{type:"range",min:"5",max:"50",step:"5",value:o,onChange:u=>c(parseInt(u.target.value,10)),style:{width:"100%",accentColor:"var(--text-primary)"}}),r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"4px"},children:[r.jsx("span",{children:"Fast (5)"}),r.jsx("span",{children:"Quality (50)"})]})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Learning Rate"}),r.jsx("input",{className:"form-input",type:"number",step:"0.00001",value:d,onChange:u=>p(parseFloat(u.target.value||"0")),style:{backgroundColor:"#0f0f0f"}})]})]}),x&&r.jsx("div",{style:{padding:"12px",backgroundColor:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.2)",borderRadius:"8px",color:"#ef4444",marginBottom:"16px",fontSize:"0.9rem"},children:x}),r.jsx("button",{className:"primary-btn",disabled:!w,onClick:f,style:{height:"48px",fontSize:"1rem",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:v?r.jsx(r.Fragment,{children:"Training..."}):r.jsxs(r.Fragment,{children:[r.jsx(xp,{size:18}),"Start Training"]})})]})}const bg=[{id:"brief",label:"Brief",description:"1-line summary"},{id:"detailed",label:"Detailed",description:"Full paragraph"},{id:"tags",label:"Tags",description:"Comma-separated keywords"},{id:"structured",label:"Structured",description:"Subject, style, mood"}],wg=[{id:"florence2",label:"Florence-2",description:"Fast & accurate (Microsoft)"},{id:"blip2",label:"BLIP-2",description:"Detailed descriptions"},{id:"cogvlm",label:"CogVLM",description:"High quality (slower)"}];function kg({onSendToPrompt:e}){const[t,n]=i.useState(null),[a,s]=i.useState(null),[l,o]=i.useState("florence2"),[c,d]=i.useState("detailed"),[p,v]=i.useState(""),[g,x]=i.useState(!1),[k,w]=i.useState(null),z=i.useCallback(y=>{var I;const j=(I=y.target.files)==null?void 0:I[0];j&&(n(j),s(URL.createObjectURL(j)),v(""),w(null))},[]),F=i.useCallback(y=>{var I;y.preventDefault();const j=(I=y.dataTransfer.files)==null?void 0:I[0];j&&j.type.startsWith("image/")&&(n(j),s(URL.createObjectURL(j)),v(""),w(null))},[]),f=async()=>{if(t){x(!0),w(null);try{const y=new FormData;y.append("file",t),y.append("model",l),y.append("mode",c);const j=await fetch(`${oe}/caption-image`,{method:"POST",body:y});if(!j.ok){const _=await j.json();throw new Error(_.detail||"Caption failed")}const I=await j.json();v(I.caption||"")}catch(y){console.error("Caption error:",y),w(y.message)}finally{x(!1)}}},u=()=>{p&&navigator.clipboard.writeText(p)},h=()=>{p&&e&&e(p)};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(gr,{size:18}),"Upload Image"]}),r.jsxs("div",{className:`upload-dropzone ${a?"has-preview":""}`,onDrop:F,onDragOver:y=>y.preventDefault(),onClick:()=>document.getElementById("i2t-file-input").click(),children:[a?r.jsx("img",{src:a,alt:"Preview",className:"upload-preview"}):r.jsxs("div",{className:"upload-placeholder",children:[r.jsx(Ye,{size:32}),r.jsx("p",{children:"Drop image here or click to upload"})]}),r.jsx("input",{id:"i2t-file-input",type:"file",accept:"image/*",onChange:z,style:{display:"none"}})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(Kt,{size:18}),"Caption Settings"]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Model"}),r.jsx("select",{value:l,onChange:y=>o(y.target.value),children:wg.map(y=>r.jsxs("option",{value:y.id,children:[y.label," - ",y.description]},y.id))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Caption Mode"}),r.jsx("div",{className:"button-group",children:bg.map(y=>r.jsx("button",{className:`btn-option ${c===y.id?"active":""}`,onClick:()=>d(y.id),title:y.description,children:y.label},y.id))})]})]}),r.jsx("button",{className:"btn-primary btn-large",onClick:f,disabled:!t||g,children:g?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Generating caption..."]}):r.jsxs(r.Fragment,{children:[r.jsx(Kt,{size:18}),"Generate Caption"]})}),k&&r.jsxs("div",{className:"error-message",children:["⚠️ ",k]}),p&&r.jsxs("div",{className:"tool-section result-section",children:[r.jsx("h3",{children:"Generated Caption"}),r.jsxs("div",{className:"caption-result",children:[r.jsx("textarea",{value:p,onChange:y=>v(y.target.value),rows:4}),r.jsxs("div",{className:"caption-actions",children:[r.jsxs("button",{className:"btn-secondary",onClick:u,children:[r.jsx(Wt,{size:16}),"Copy"]}),e&&r.jsxs("button",{className:"btn-primary",onClick:h,children:[r.jsx(pp,{size:16}),"Use as Prompt"]})]})]})]}),r.jsx("style",{children:`
        .upload-dropzone {
          border: 2px dashed var(--border-color, #444);
          border-radius: 12px;
          padding: 40px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
          min-height: 200px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .upload-dropzone:hover {
          border-color: var(--accent-color, #7c3aed);
          background: rgba(124, 58, 237, 0.05);
        }
        .upload-dropzone.has-preview {
          padding: 8px;
        }
        .upload-preview {
          max-width: 100%;
          max-height: 300px;
          border-radius: 8px;
          object-fit: contain;
        }
        .upload-placeholder {
          color: var(--text-muted, #888);
        }
        .upload-placeholder p {
          margin-top: 12px;
        }
        .button-group {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }
        .btn-option {
          padding: 8px 16px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: transparent;
          color: var(--text-color, #fff);
          cursor: pointer;
          transition: all 0.2s;
        }
        .btn-option:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .btn-option.active {
          background: var(--accent-color, #7c3aed);
          border-color: var(--accent-color, #7c3aed);
        }
        .caption-result textarea {
          width: 100%;
          padding: 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-family: inherit;
          resize: vertical;
        }
        .caption-actions {
          display: flex;
          gap: 8px;
          margin-top: 12px;
        }
        .error-message {
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          margin-top: 12px;
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `})]})}const Oc=[{id:"cinematic",label:"🎬 Cinematic",keywords:"cinematic lighting, film grain, dramatic shadows, professional photography"},{id:"anime",label:"🎌 Anime",keywords:"anime style, vibrant colors, cel shading, Japanese animation"},{id:"photorealistic",label:"📸 Photorealistic",keywords:"photorealistic, highly detailed, 8k, sharp focus, professional photo"},{id:"abstract",label:"🎨 Abstract",keywords:"abstract art, geometric shapes, vibrant colors, artistic"},{id:"vintage",label:"📼 Vintage",keywords:"vintage aesthetic, retro, film photography, nostalgic, 1970s"},{id:"cyberpunk",label:"🤖 Cyberpunk",keywords:"cyberpunk, neon lights, futuristic, dystopian, high tech low life"},{id:"fantasy",label:"🧙 Fantasy",keywords:"fantasy art, magical, ethereal lighting, mystical, enchanted"},{id:"minimalist",label:"⬜ Minimalist",keywords:"minimalist, clean, simple, negative space, modern"},{id:"horror",label:"👻 Horror",keywords:"dark atmosphere, eerie, horror, unsettling, creepy"},{id:"scifi",label:"🚀 Sci-Fi",keywords:"science fiction, futuristic, space, advanced technology"}];function Sg({onSendToTool:e}){const[t,n]=i.useState(""),[a,s]=i.useState(""),[l,o]=i.useState("expand"),[c,d]=i.useState(!0),[p,v]=i.useState(!1),[g,x]=i.useState(null),[k,w]=i.useState(!1),[z,F]=i.useState(null),f=async()=>{if(t.trim()){w(!0),F(null);try{const y=await fetch(`${oe}/generate-prompt`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({input:t.trim(),style:a||null,mode:l,include_negative:c,include_motion:p})});if(!y.ok){const I=await y.json();throw new Error(I.detail||"Generation failed")}const j=await y.json();x(j)}catch(y){console.error("Prompt generation error:",y),F(y.message)}finally{w(!1)}}},u=()=>{if(!t.trim())return;const y=t.trim(),j=Oc.find(W=>W.id===a),I=j?`, ${j.keywords}`:"",_=`${y}${I}, masterpiece, best quality, highly detailed`;x({prompt:_,negative_prompt:c?"ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text, cropped, worst quality":"",motion_prompt:p?"smooth camera motion, cinematic movement, fluid animation":"",variations:null})},h=y=>{navigator.clipboard.writeText(y)};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(Gt,{size:18}),"Input Idea"]}),r.jsx("textarea",{value:t,onChange:y=>n(y.target.value),placeholder:"Describe your image or video idea... (e.g., 'a cat wearing sunglasses')",rows:3,className:"prompt-input"})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Style Preset"}),r.jsx("div",{className:"style-grid",children:Oc.map(y=>r.jsx("button",{className:`style-btn ${a===y.id?"active":""}`,onClick:()=>s(a===y.id?"":y.id),children:y.label},y.id))})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Options"}),r.jsxs("div",{className:"options-row",children:[r.jsxs("label",{className:"checkbox-label",children:[r.jsx("input",{type:"checkbox",checked:c,onChange:y=>d(y.target.checked)}),"Generate negative prompt"]}),r.jsxs("label",{className:"checkbox-label",children:[r.jsx("input",{type:"checkbox",checked:p,onChange:y=>v(y.target.checked)}),"Include motion prompts (for video)"]})]})]}),r.jsxs("div",{className:"button-row",children:[r.jsxs("button",{className:"btn-primary btn-large",onClick:u,disabled:!t.trim(),children:[r.jsx(Kt,{size:18}),"Quick Generate"]}),r.jsx("button",{className:"btn-secondary btn-large",onClick:f,disabled:!t.trim()||k,title:"Uses AI for smarter enhancement (requires LLM)",children:k?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Generating..."]}):r.jsxs(r.Fragment,{children:[r.jsx(Gt,{size:18}),"AI Enhance"]})})]}),z&&r.jsxs("div",{className:"error-message",children:["⚠️ ",z]}),g&&r.jsxs("div",{className:"results-section",children:[r.jsxs("div",{className:"result-card",children:[r.jsxs("div",{className:"result-header",children:[r.jsx("h4",{children:"✨ Enhanced Prompt"}),r.jsx("button",{className:"btn-icon",onClick:()=>h(g.prompt),children:r.jsx(Wt,{size:16})})]}),r.jsx("p",{className:"result-text",children:g.prompt})]}),g.negative_prompt&&r.jsxs("div",{className:"result-card",children:[r.jsxs("div",{className:"result-header",children:[r.jsx("h4",{children:"🚫 Negative Prompt"}),r.jsx("button",{className:"btn-icon",onClick:()=>h(g.negative_prompt),children:r.jsx(Wt,{size:16})})]}),r.jsx("p",{className:"result-text muted",children:g.negative_prompt})]}),g.motion_prompt&&r.jsxs("div",{className:"result-card",children:[r.jsxs("div",{className:"result-header",children:[r.jsx("h4",{children:"🎬 Motion Prompt"}),r.jsx("button",{className:"btn-icon",onClick:()=>h(g.motion_prompt),children:r.jsx(Wt,{size:16})})]}),r.jsx("p",{className:"result-text",children:g.motion_prompt})]}),g.variations&&g.variations.length>0&&r.jsxs("div",{className:"result-card",children:[r.jsx("h4",{children:"🔄 Variations"}),g.variations.map((y,j)=>r.jsxs("div",{className:"variation-item",children:[r.jsx("p",{className:"result-text",children:y}),r.jsx("button",{className:"btn-icon",onClick:()=>h(y),children:r.jsx(Wt,{size:16})})]},j))]}),e&&r.jsxs("button",{className:"btn-primary",onClick:()=>e(g),children:[r.jsx(pp,{size:16}),"Send to Generator"]})]}),r.jsx("style",{children:`
        .prompt-input {
          width: 100%;
          padding: 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-family: inherit;
          font-size: 14px;
          resize: vertical;
        }
        .style-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
          gap: 8px;
        }
        .style-btn {
          padding: 10px 12px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: transparent;
          color: var(--text-color, #fff);
          cursor: pointer;
          transition: all 0.2s;
          font-size: 13px;
        }
        .style-btn:hover {
          border-color: var(--accent-color, #7c3aed);
          background: rgba(124, 58, 237, 0.1);
        }
        .style-btn.active {
          background: var(--accent-color, #7c3aed);
          border-color: var(--accent-color, #7c3aed);
        }
        .options-row {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .checkbox-label {
          display: flex;
          align-items: center;
          gap: 8px;
          cursor: pointer;
        }
        .checkbox-label input {
          width: 16px;
          height: 16px;
        }
        .button-row {
          display: flex;
          gap: 12px;
          margin-top: 16px;
        }
        .btn-large {
          flex: 1;
          padding: 14px 20px;
          font-size: 15px;
        }
        .results-section {
          margin-top: 24px;
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .result-card {
          background: var(--bg-secondary, #1a1a1a);
          border: 1px solid var(--border-color, #444);
          border-radius: 12px;
          padding: 16px;
        }
        .result-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }
        .result-header h4 {
          margin: 0;
          font-size: 14px;
        }
        .result-text {
          margin: 0;
          line-height: 1.5;
          word-break: break-word;
        }
        .result-text.muted {
          color: var(--text-muted, #888);
        }
        .variation-item {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 12px;
          padding: 8px 0;
          border-bottom: 1px solid var(--border-color, #333);
        }
        .variation-item:last-child {
          border-bottom: none;
        }
        .btn-icon {
          background: none;
          border: none;
          color: var(--text-muted, #888);
          cursor: pointer;
          padding: 4px;
          border-radius: 4px;
        }
        .btn-icon:hover {
          color: var(--text-color, #fff);
          background: var(--bg-hover, #333);
        }
        .error-message {
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          margin-top: 12px;
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `})]})}const Ac=[{value:"CyberRealistic_Pony_v14.1_FP16.safetensors",label:"CyberRealistic Pony"},{value:"dreamshaperXL_lightningDPMSDE.safetensors",label:"Dreamshaper Lightning"},{value:"juggernautXL_ragnarok.safetensors",label:"Juggernaut XL"},{value:"waiIllustriousSDXL_v160.safetensors",label:"Wai Illustrious (Anime)"}];function Ng({onOutput:e,onJobSubmitted:t}){const[n,a]=i.useState(null),[s,l]=i.useState(null),[o,c]=i.useState(""),[d,p]=i.useState("ugly, deformed, blurry, low quality, bad anatomy, watermark"),[v,g]=i.useState(.6),[x,k]=i.useState("CyberRealistic_Pony_v14.1_FP16.safetensors"),[w,z]=i.useState(!1),[F,f]=i.useState(25),[u,h]=i.useState(7),[y,j]=i.useState(-1),[I,_]=i.useState("dpmpp_2m"),[R,G]=i.useState("karras"),[W,b]=i.useState(!1),[N,L]=i.useState(null),[ee,T]=i.useState(null),[ne,ae]=i.useState(null),D=i.useCallback(V=>{var Q;const H=(Q=V.target.files)==null?void 0:Q[0];H&&(a(H),l(URL.createObjectURL(H)),ae(null),L(null),T(null))},[]),U=i.useCallback(V=>{var Q;V.preventDefault();const H=(Q=V.dataTransfer.files)==null?void 0:Q[0];H&&H.type.startsWith("image/")&&(a(H),l(URL.createObjectURL(H)),ae(null),L(null),T(null))},[]),q=async()=>{var V,H,Q;if(n){b(!0),L(null),T(null);try{const C=new FormData;C.append("file",n),C.append("prompt",o||"high quality, detailed"),C.append("negative_prompt",d),C.append("denoise",String(v)),C.append("checkpoint",x),C.append("steps",String(F)),C.append("cfg",String(u)),C.append("seed",String(y)),C.append("sampler_name",I),C.append("scheduler",R);const Y=await We(`${oe}/generate-i2i`,C);if(!Y.ok)throw new Error(((V=Y.data)==null?void 0:V.detail)||"Generation failed");const M=(H=Y.data)==null?void 0:H.prompt_id;if(!M)throw new Error("No prompt_id returned");T({promptId:M,checkpoint:((Q=Ac.find(m=>m.value===x))==null?void 0:Q.label)||x}),t&&t({prompt_id:M})}catch(C){console.error("I2I error:",C),L(C.message)}finally{b(!1)}}};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(gr,{size:18}),"Source Image"]}),r.jsxs("div",{className:`upload-dropzone ${s?"has-preview":""}`,onDrop:U,onDragOver:V=>V.preventDefault(),onClick:()=>document.getElementById("i2i-file-input").click(),children:[s?r.jsx("img",{src:s,alt:"Preview",className:"upload-preview"}):r.jsxs("div",{className:"upload-placeholder",children:[r.jsx(Ye,{size:32}),r.jsx("p",{children:"Drop image here or click to upload"})]}),r.jsx("input",{id:"i2i-file-input",type:"file",accept:"image/*",onChange:D,style:{display:"none"}})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(Kt,{size:18}),"Transformation"]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Prompt (describe desired changes)"}),r.jsx("textarea",{value:o,onChange:V=>c(V.target.value),rows:3,placeholder:"Describe what you want the image to become... (e.g., 'anime style illustration')"})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("label",{children:[r.jsx(pa,{size:14}),"Denoise Strength",r.jsx("span",{className:"label-value",children:v.toFixed(2)})]}),r.jsx("input",{type:"range",min:"0.1",max:"1.0",step:"0.05",value:v,onChange:V=>g(parseFloat(V.target.value))}),r.jsxs("div",{className:"range-labels",children:[r.jsx("span",{children:"Subtle (0.1)"}),r.jsx("span",{children:"Complete (1.0)"})]}),r.jsxs("div",{className:"denoise-hint",children:[v<.3&&"💡 Minor adjustments, preserves most of original",v>=.3&&v<.6&&"💡 Moderate changes, good balance",v>=.6&&v<.8&&"💡 Significant transformation",v>=.8&&"💡 Near-complete regeneration from prompt"]})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Model"}),r.jsx("select",{value:x,onChange:V=>k(V.target.value),children:Ac.map(V=>r.jsx("option",{value:V.value,children:V.label},V.value))})]})]}),r.jsxs("div",{className:"tool-section collapsible",children:[r.jsxs("button",{className:"section-toggle",onClick:()=>z(!w),children:[r.jsx(vr,{size:16}),"Advanced Settings",r.jsx(Tt,{size:16,className:w?"rotated":""})]}),w&&r.jsxs("div",{className:"advanced-content",children:[r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Negative Prompt"}),r.jsx("textarea",{value:d,onChange:V=>p(V.target.value),rows:2})]}),r.jsxs("div",{className:"form-row",children:[r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"Steps"}),r.jsx("input",{type:"number",value:F,onChange:V=>f(parseInt(V.target.value)||25),min:"1",max:"50"})]}),r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"CFG Scale"}),r.jsx("input",{type:"number",value:u,onChange:V=>h(parseFloat(V.target.value)||7),min:"1",max:"20",step:"0.5"})]})]}),r.jsxs("div",{className:"form-row",children:[r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"Sampler"}),r.jsxs("select",{value:I,onChange:V=>_(V.target.value),children:[r.jsx("option",{value:"euler",children:"Euler"}),r.jsx("option",{value:"euler_ancestral",children:"Euler Ancestral"}),r.jsx("option",{value:"dpmpp_2m",children:"DPM++ 2M"}),r.jsx("option",{value:"dpmpp_2m_sde",children:"DPM++ 2M SDE"}),r.jsx("option",{value:"dpmpp_3m_sde",children:"DPM++ 3M SDE"})]})]}),r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"Scheduler"}),r.jsxs("select",{value:R,onChange:V=>G(V.target.value),children:[r.jsx("option",{value:"normal",children:"Normal"}),r.jsx("option",{value:"karras",children:"Karras"}),r.jsx("option",{value:"exponential",children:"Exponential"}),r.jsx("option",{value:"sgm_uniform",children:"SGM Uniform"})]})]})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Seed (-1 = random)"}),r.jsx("input",{type:"number",value:y,onChange:V=>j(parseInt(V.target.value)||-1)})]})]})]}),ee&&r.jsxs("div",{className:"queued-notice",children:["✅ Job queued! Check the Queue panel for progress.",r.jsx("span",{className:"queued-mode",children:ee.checkpoint})]}),N&&r.jsxs("div",{className:"error-message",children:["⚠️ ",N]}),r.jsx("button",{className:"btn-primary btn-large",onClick:q,disabled:!n||W,children:W?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Queueing..."]}):r.jsxs(r.Fragment,{children:[r.jsx(Kt,{size:18}),"Transform Image"]})}),ne&&r.jsxs("div",{className:"result-section",children:[r.jsx("h3",{children:"Result"}),r.jsxs("div",{className:"comparison",children:[r.jsxs("div",{className:"comparison-item",children:[r.jsx("span",{className:"comparison-label",children:"Original"}),r.jsx("img",{src:s,alt:"Original"})]}),r.jsxs("div",{className:"comparison-item",children:[r.jsx("span",{className:"comparison-label",children:"Transformed"}),r.jsx("img",{src:ne,alt:"Result"})]})]})]}),r.jsx("style",{children:`
        .upload-dropzone {
          border: 2px dashed var(--border-color, #444);
          border-radius: 12px;
          padding: 40px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
          min-height: 200px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .upload-dropzone:hover {
          border-color: var(--accent-color, #7c3aed);
          background: rgba(124, 58, 237, 0.05);
        }
        .upload-dropzone.has-preview {
          padding: 8px;
        }
        .upload-preview {
          max-width: 100%;
          max-height: 300px;
          border-radius: 8px;
          object-fit: contain;
        }
        .upload-placeholder {
          color: var(--text-muted, #888);
        }
        .upload-placeholder p {
          margin-top: 12px;
        }
        .form-group {
          margin-bottom: 16px;
        }
        .form-group label {
          display: flex;
          align-items: center;
          gap: 6px;
          margin-bottom: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .label-value {
          margin-left: auto;
          color: var(--accent-color, #7c3aed);
          font-weight: 500;
        }
        .form-group textarea,
        .form-group select,
        .form-group input[type="number"] {
          width: 100%;
          padding: 10px 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 13px;
        }
        .range-labels {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: var(--text-muted, #888);
          margin-top: 4px;
        }
        .denoise-hint {
          margin-top: 8px;
          font-size: 12px;
          color: var(--text-muted, #888);
        }
        .form-row {
          display: flex;
          gap: 16px;
        }
        .form-group.half {
          flex: 1;
        }
        .section-toggle {
          display: flex;
          align-items: center;
          gap: 8px;
          width: 100%;
          padding: 12px;
          background: transparent;
          border: 1px solid var(--border-color, #333);
          border-radius: 8px;
          color: var(--text-secondary, #aaa);
          cursor: pointer;
          font-size: 13px;
        }
        .section-toggle:hover {
          border-color: var(--border-color, #555);
        }
        .section-toggle .rotated {
          transform: rotate(180deg);
        }
        .section-toggle svg:last-child {
          margin-left: auto;
          transition: transform 0.2s;
        }
        .advanced-content {
          margin-top: 16px;
          padding-top: 16px;
          border-top: 1px solid var(--border-color, #333);
        }
        .progress-section {
          margin: 16px 0;
        }
        .progress-bar {
          height: 4px;
          background: var(--bg-secondary, #333);
          border-radius: 2px;
          overflow: hidden;
        }
        .progress-fill {
          height: 100%;
          background: var(--accent-color, #7c3aed);
          transition: width 0.3s;
        }
        .progress-status {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .error-message {
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          margin: 12px 0;
        }
        .result-section {
          margin-top: 24px;
          padding-top: 24px;
          border-top: 1px solid var(--border-color, #333);
        }
        .comparison {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
          margin-top: 16px;
        }
        .comparison-item {
          position: relative;
        }
        .comparison-label {
          position: absolute;
          top: 8px;
          left: 8px;
          background: rgba(0,0,0,0.7);
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 11px;
        }
        .comparison-item img {
          width: 100%;
          border-radius: 8px;
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `})]})}const wl=[{value:"RealESRGAN_x4plus.pth",label:"RealESRGAN 4x (General)",scale:4},{value:"RealESRGAN_x4plus_anime_6B.pth",label:"RealESRGAN 4x (Anime)",scale:4},{value:"RealESRGAN_x2plus.pth",label:"RealESRGAN 2x",scale:2},{value:"4x-UltraSharp.pth",label:"4x UltraSharp",scale:4},{value:"4x_NMKD-Siax_200k.pth",label:"4x NMKD-Siax",scale:4}],Cg=[2,4];function _g({onOutput:e,onJobSubmitted:t}){const[n,a]=i.useState(null),[s,l]=i.useState(null),[o,c]=i.useState(null),[d,p]=i.useState("RealESRGAN_x4plus.pth"),[v,g]=i.useState(4),[x,k]=i.useState(!1),[w,z]=i.useState(!1),[F,f]=i.useState(null),[u,h]=i.useState(null),[y,j]=i.useState(null),I=i.useCallback(b=>{var L;const N=(L=b.target.files)==null?void 0:L[0];if(N){a(N);const ee=URL.createObjectURL(N);l(ee),j(null),f(null),h(null);const T=new Image;T.onload=()=>{c({width:T.width,height:T.height})},T.src=ee}},[]),_=i.useCallback(b=>{var L;b.preventDefault();const N=(L=b.dataTransfer.files)==null?void 0:L[0];if(N&&N.type.startsWith("image/")){a(N);const ee=URL.createObjectURL(N);l(ee),j(null),f(null),h(null);const T=new Image;T.onload=()=>{c({width:T.width,height:T.height})},T.src=ee}},[]),R=async()=>{var b,N,L;if(n){z(!0),f(null),h(null);try{const ee=new FormData;ee.append("file",n),ee.append("model",d),ee.append("scale",String(v)),ee.append("face_enhance",String(x));const T=await We(`${oe}/upscale`,ee);if(!T.ok)throw new Error(((b=T.data)==null?void 0:b.detail)||"Upscaling failed");const ne=(N=T.data)==null?void 0:N.prompt_id;if(!ne)throw new Error("No prompt_id returned");h({promptId:ne,model:((L=wl.find(ae=>ae.value===d))==null?void 0:L.label)||d,scale:v}),t&&t({prompt_id:ne})}catch(ee){console.error("Upscale error:",ee),f(ee.message)}finally{z(!1)}}};wl.find(b=>b.value===d);const G=o?o.width*v:0,W=o?o.height*v:0;return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(gr,{size:18}),"Source Image"]}),r.jsxs("div",{className:`upload-dropzone ${s?"has-preview":""}`,onDrop:_,onDragOver:b=>b.preventDefault(),onClick:()=>document.getElementById("upscale-file-input").click(),children:[s?r.jsx("img",{src:s,alt:"Preview",className:"upload-preview"}):r.jsxs("div",{className:"upload-placeholder",children:[r.jsx(Ye,{size:32}),r.jsx("p",{children:"Drop image here or click to upload"})]}),r.jsx("input",{id:"upscale-file-input",type:"file",accept:"image/*",onChange:I,style:{display:"none"}})]}),o&&r.jsxs("div",{className:"image-info",children:[r.jsxs("span",{children:["📐 ",o.width," × ",o.height,"px"]}),r.jsx("span",{children:"→"}),r.jsxs("span",{className:"output-size",children:[G," × ",W,"px"]})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(Fc,{size:18}),"Upscale Settings"]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Scale Factor"}),r.jsx("div",{className:"button-group",children:Cg.map(b=>r.jsxs("button",{className:`btn-option ${v===b?"active":""}`,onClick:()=>g(b),type:"button",children:[b,"x"]},b))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Upscale Model"}),r.jsx("select",{value:d,onChange:b=>p(b.target.value),children:wl.map(b=>r.jsx("option",{value:b.value,children:b.label},b.value))})]}),r.jsx("div",{className:"form-group",children:r.jsxs("label",{className:"checkbox-label",children:[r.jsx("input",{type:"checkbox",checked:x,onChange:b=>k(b.target.checked)}),"Face Enhancement (GFPGAN)",r.jsx("span",{className:"hint",children:"Improves face details"})]})})]}),u&&r.jsxs("div",{className:"queued-notice",children:["✅ Job queued! Check the Queue panel for progress.",r.jsxs("span",{className:"queued-mode",children:[u.scale,"x ",u.model]})]}),F&&r.jsxs("div",{className:"error-message",children:["⚠️ ",F]}),r.jsx("button",{className:"btn-primary btn-large",onClick:R,disabled:!n||w,children:w?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Queueing..."]}):r.jsxs(r.Fragment,{children:[r.jsx(Fc,{size:18}),"Upscale Image"]})}),y&&r.jsxs("div",{className:"result-section",children:[r.jsxs("h3",{children:["Result (",v,"x Upscaled)"]}),r.jsx("div",{className:"result-image",children:r.jsx("img",{src:y,alt:"Upscaled"})}),r.jsx("a",{href:y,download:!0,className:"btn-secondary",style:{marginTop:12,display:"inline-flex",alignItems:"center",gap:8},children:"Download Full Resolution"})]}),r.jsx("style",{children:`
        .upload-dropzone {
          border: 2px dashed var(--border-color, #444);
          border-radius: 12px;
          padding: 40px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
          min-height: 200px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .upload-dropzone:hover {
          border-color: var(--accent-color, #7c3aed);
          background: rgba(124, 58, 237, 0.05);
        }
        .upload-dropzone.has-preview {
          padding: 8px;
        }
        .upload-preview {
          max-width: 100%;
          max-height: 300px;
          border-radius: 8px;
          object-fit: contain;
        }
        .upload-placeholder {
          color: var(--text-muted, #888);
        }
        .upload-placeholder p {
          margin-top: 12px;
        }
        .image-info {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 12px;
          margin-top: 12px;
          font-size: 13px;
          color: var(--text-muted, #888);
        }
        .output-size {
          color: var(--accent-color, #7c3aed);
          font-weight: 500;
        }
        .form-group {
          margin-bottom: 16px;
        }
        .form-group label {
          display: block;
          margin-bottom: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .form-group select {
          width: 100%;
          padding: 10px 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 13px;
        }
        .button-group {
          display: flex;
          gap: 8px;
        }
        .btn-option {
          padding: 12px 24px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: transparent;
          color: var(--text-color, #fff);
          cursor: pointer;
          transition: all 0.2s;
          font-size: 15px;
          font-weight: 500;
        }
        .btn-option:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .btn-option.active {
          background: var(--accent-color, #7c3aed);
          border-color: var(--accent-color, #7c3aed);
        }
        .checkbox-label {
          display: flex !important;
          align-items: center;
          gap: 8px;
          cursor: pointer;
        }
        .checkbox-label input {
          width: 16px;
          height: 16px;
        }
        .checkbox-label .hint {
          margin-left: auto;
          font-size: 12px;
          color: var(--text-muted, #666);
        }
        .progress-section {
          margin: 16px 0;
        }
        .progress-bar {
          height: 4px;
          background: var(--bg-secondary, #333);
          border-radius: 2px;
          overflow: hidden;
        }
        .progress-fill {
          height: 100%;
          background: var(--accent-color, #7c3aed);
          transition: width 0.3s;
        }
        .progress-status {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .error-message {
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          margin: 12px 0;
        }
        .result-section {
          margin-top: 24px;
          padding-top: 24px;
          border-top: 1px solid var(--border-color, #333);
        }
        .result-image img {
          width: 100%;
          max-height: 400px;
          object-fit: contain;
          border-radius: 8px;
          margin-top: 12px;
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `})]})}const $c=[{value:"nova",label:"Nova",desc:"Friendly, upbeat",gender:"female"},{value:"shimmer",label:"Shimmer",desc:"Soft, gentle",gender:"female"},{value:"alloy",label:"Alloy",desc:"Neutral, versatile",gender:"female"},{value:"echo",label:"Echo",desc:"Warm, conversational",gender:"male"},{value:"fable",label:"Fable",desc:"Expressive, dramatic",gender:"male"},{value:"onyx",label:"Onyx",desc:"Deep, authoritative",gender:"male"}],zg=[{value:"tts",label:"Text to Speech",icon:r.jsx(gi,{size:18}),desc:"Generate voice from text"},{value:"music",label:"Music Generation",icon:r.jsx(dx,{size:18}),desc:"Generate music/sounds"},{value:"sfx",label:"Sound Effects",icon:r.jsx(hn,{size:18}),desc:"Generate sound effects"}],Eg=[{value:"ambient",label:"Ambient"},{value:"cinematic",label:"Cinematic"},{value:"electronic",label:"Electronic"},{value:"jazz",label:"Jazz"},{value:"classical",label:"Classical"},{value:"lofi",label:"Lo-Fi"},{value:"rock",label:"Rock"},{value:"hiphop",label:"Hip-Hop"}];function Tg({onOutput:e,onJobSubmitted:t}){const[n,a]=i.useState("tts"),[s,l]=i.useState(""),[o,c]=i.useState("nova"),[d,p]=i.useState("cinematic"),[v,g]=i.useState(10),[x,k]=i.useState(!1),[w,z]=i.useState(1),[F,f]=i.useState(1),[u,h]=i.useState(!1),[y,j]=i.useState(null),[I,_]=i.useState(null),[R,G]=i.useState(null),[W,b]=i.useState(!1),N=i.useRef(null),L=async()=>{var T,ne,ae;if(s.trim()){h(!0),j(null),_(null);try{let D="/generate-audio";const U=new FormData;U.append("text",s.trim()),U.append("mode",n),n==="tts"?(U.append("voice",o),U.append("speed",w.toString()),U.append("pitch",F.toString())):n==="music"?(U.append("style",d),U.append("duration",v.toString())):n==="sfx"&&U.append("duration",Math.min(v,10).toString());const q=await We(`${oe}${D}`,U);if(!q.ok){const V=typeof q.data=="object"?((T=q.data)==null?void 0:T.detail)||JSON.stringify(q.data):q.data||"Audio generation failed";throw new Error(V)}if((ne=q.data)!=null&&ne.prompt_id)_({promptId:q.data.prompt_id,mode:n,text:s.substring(0,50)+(s.length>50?"...":"")}),t&&t(q.data);else if((ae=q.data)!=null&&ae.url){const V=q.data.url,H=V.startsWith("http")?V:`${oe}${V}`;G({url:H,filename:V.split("/").pop()}),e&&e({kind:"audio",url:H,filename:V.split("/").pop()})}}catch(D){console.error("Audio error:",D),j(D.message)}finally{h(!1)}}},ee=()=>{N.current&&(W?N.current.pause():N.current.play(),b(!W))};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(hn,{size:18}),"Generation Mode"]}),r.jsx("div",{className:"mode-grid",children:zg.map(T=>r.jsxs("button",{className:`mode-btn ${n===T.value?"active":""}`,onClick:()=>a(T.value),children:[T.icon,r.jsx("span",{className:"mode-name",children:T.label}),r.jsx("span",{className:"mode-desc",children:T.desc})]},T.value))})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:n==="tts"?"Text to Speak":n==="music"?"Music Prompt":"Sound Description"}),r.jsx("textarea",{value:s,onChange:T=>l(T.target.value),placeholder:n==="tts"?"Enter the text you want to convert to speech...":n==="music"?'Describe the music you want to generate (e.g., "upbeat electronic dance track with heavy bass")':'Describe the sound effect (e.g., "thunder rumbling in the distance")',rows:4,className:"prompt-textarea"})]}),n==="tts"&&r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Voice"}),r.jsxs("div",{className:"voice-group",children:[r.jsx("span",{className:"voice-group-label",children:"Female"}),r.jsx("div",{className:"voice-grid",children:$c.filter(T=>T.gender==="female").map(T=>r.jsxs("button",{className:`voice-btn ${o===T.value?"active":""}`,onClick:()=>c(T.value),children:[r.jsx("span",{className:"voice-name",children:T.label}),r.jsx("span",{className:"voice-desc",children:T.desc})]},T.value))})]}),r.jsxs("div",{className:"voice-group",children:[r.jsx("span",{className:"voice-group-label",children:"Male"}),r.jsx("div",{className:"voice-grid",children:$c.filter(T=>T.gender==="male").map(T=>r.jsxs("button",{className:`voice-btn ${o===T.value?"active":""}`,onClick:()=>c(T.value),children:[r.jsx("span",{className:"voice-name",children:T.label}),r.jsx("span",{className:"voice-desc",children:T.desc})]},T.value))})]})]}),n==="music"&&r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Style"}),r.jsx("div",{className:"style-grid",children:Eg.map(T=>r.jsx("button",{className:`style-btn ${d===T.value?"active":""}`,onClick:()=>p(T.value),children:T.label},T.value))})]}),(n==="music"||n==="sfx")&&r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Duration"}),r.jsxs("div",{className:"slider-row",children:[r.jsx("input",{type:"range",min:n==="sfx"?1:5,max:n==="sfx"?10:30,value:v,onChange:T=>g(parseInt(T.target.value))}),r.jsxs("span",{className:"slider-value",children:[v,"s"]})]})]}),n==="tts"&&r.jsxs("div",{className:"tool-section collapsible",children:[r.jsxs("h3",{onClick:()=>k(!x),style:{cursor:"pointer"},children:[r.jsx(vr,{size:16}),"Advanced",r.jsx(Tt,{size:16,style:{marginLeft:"auto",transform:x?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.2s"}})]}),x&&r.jsxs("div",{className:"advanced-content",children:[r.jsxs("div",{className:"slider-row",children:[r.jsx("label",{children:"Speed"}),r.jsx("input",{type:"range",min:.5,max:2,step:.1,value:w,onChange:T=>z(parseFloat(T.target.value))}),r.jsxs("span",{className:"slider-value",children:[w.toFixed(1),"x"]})]}),r.jsxs("div",{className:"slider-row",children:[r.jsx("label",{children:"Pitch"}),r.jsx("input",{type:"range",min:.5,max:2,step:.1,value:F,onChange:T=>f(parseFloat(T.target.value))}),r.jsxs("span",{className:"slider-value",children:[F.toFixed(1),"x"]})]})]})]}),I&&r.jsxs("div",{className:"queued-notice",children:["✅ Job queued! Check the Queue panel for progress.",r.jsx("span",{className:"queued-mode",children:I.mode.toUpperCase()})]}),y&&r.jsxs("div",{className:"error-message",children:["⚠️ ",y]}),r.jsx("button",{className:"btn-primary btn-large",onClick:L,disabled:!s.trim()||u,children:u?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Queueing..."]}):r.jsxs(r.Fragment,{children:[r.jsx(hn,{size:18}),"Generate ",n==="tts"?"Speech":n==="music"?"Music":"Sound"]})}),R&&r.jsxs("div",{className:"result-section",children:[r.jsx("h3",{children:"Result"}),r.jsxs("div",{className:"audio-player",children:[r.jsx("audio",{ref:N,src:R.url,onEnded:()=>b(!1),onPlay:()=>b(!0),onPause:()=>b(!1)}),r.jsx("button",{className:"play-btn",onClick:ee,children:W?r.jsx(yo,{size:24}):r.jsx(ua,{size:24})}),r.jsx("div",{className:"audio-info",children:r.jsx("span",{className:"audio-filename",children:R.filename})}),r.jsx("a",{href:R.url,download:!0,className:"download-btn",children:r.jsx(vt,{size:18})})]})]}),r.jsx("style",{children:`
        .tool-section {
          margin-bottom: 20px;
        }
        .tool-section h3 {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          font-weight: 500;
          margin-bottom: 12px;
          color: var(--text-color, #fff);
        }
        .mode-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }
        .mode-btn {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 6px;
          padding: 16px 12px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: var(--bg-secondary, #1a1a1a);
          cursor: pointer;
          transition: all 0.2s;
        }
        .mode-btn:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .mode-btn.active {
          background: rgba(124, 58, 237, 0.2);
          border-color: var(--accent-color, #7c3aed);
        }
        .mode-name {
          font-size: 13px;
          font-weight: 500;
          color: var(--text-color, #fff);
        }
        .mode-desc {
          font-size: 10px;
          color: var(--text-muted, #888);
          text-align: center;
        }
        .prompt-textarea {
          width: 100%;
          padding: 12px;
          border-radius: 8px;
          border: 1px solid var(--border-color, #444);
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 14px;
          resize: none;
        }
        .voice-group {
          margin-bottom: 12px;
        }
        .voice-group:last-child {
          margin-bottom: 0;
        }
        .voice-group-label {
          display: block;
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          color: var(--text-muted, #888);
          margin-bottom: 8px;
        }
        .voice-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }
        .voice-btn {
          padding: 10px;
          border: 1px solid var(--border-color, #444);
          border-radius: 8px;
          background: var(--bg-secondary, #1a1a1a);
          cursor: pointer;
          transition: all 0.2s;
          text-align: left;
        }
        .voice-btn:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .voice-btn.active {
          background: rgba(124, 58, 237, 0.2);
          border-color: var(--accent-color, #7c3aed);
        }
        .voice-name {
          display: block;
          font-size: 13px;
          font-weight: 500;
          color: var(--text-color, #fff);
        }
        .voice-desc {
          display: block;
          font-size: 10px;
          color: var(--text-muted, #888);
        }
        .style-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 6px;
        }
        .style-btn {
          padding: 8px 12px;
          border: 1px solid var(--border-color, #444);
          border-radius: 6px;
          background: var(--bg-secondary, #1a1a1a);
          color: var(--text-color, #fff);
          font-size: 12px;
          cursor: pointer;
          transition: all 0.2s;
        }
        .style-btn:hover {
          border-color: var(--accent-color, #7c3aed);
        }
        .style-btn.active {
          background: var(--accent-color, #7c3aed);
          border-color: var(--accent-color, #7c3aed);
        }
        .slider-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .slider-row label {
          min-width: 60px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .slider-row input[type="range"] {
          flex: 1;
        }
        .slider-value {
          min-width: 45px;
          text-align: right;
          font-weight: 500;
          color: var(--accent-color, #7c3aed);
        }
        .collapsible h3 {
          padding: 12px;
          margin: -12px -12px 0;
          border-radius: 8px;
        }
        .collapsible h3:hover {
          background: var(--bg-secondary, #1a1a1a);
        }
        .advanced-content {
          margin-top: 12px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .progress-section {
          margin: 16px 0;
        }
        .progress-bar {
          height: 4px;
          background: var(--bg-secondary, #333);
          border-radius: 2px;
          overflow: hidden;
        }
        .progress-fill {
          height: 100%;
          background: var(--accent-color, #7c3aed);
          transition: width 0.3s;
        }
        .progress-status {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 8px;
          font-size: 13px;
          color: var(--text-secondary, #aaa);
        }
        .error-message {
          padding: 12px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          margin: 12px 0;
        }
        .result-section {
          margin-top: 24px;
          padding-top: 24px;
          border-top: 1px solid var(--border-color, #333);
        }
        .audio-player {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 16px;
          background: var(--bg-secondary, #1a1a1a);
          border-radius: 12px;
        }
        .play-btn {
          width: 48px;
          height: 48px;
          border-radius: 50%;
          border: none;
          background: var(--accent-color, #7c3aed);
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: transform 0.2s;
        }
        .play-btn:hover {
          transform: scale(1.05);
        }
        .audio-info {
          flex: 1;
        }
        .audio-filename {
          font-size: 13px;
          color: var(--text-color, #fff);
        }
        .download-btn {
          padding: 8px;
          border-radius: 6px;
          background: var(--bg-tertiary, #252525);
          color: var(--text-color, #fff);
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .download-btn:hover {
          background: var(--border-color, #444);
        }
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `})]})}const Pg=["audio/wav","audio/mp3","audio/mpeg","audio/flac","audio/ogg","audio/webm"],Uc=[{value:"F5v1",label:"F5 v1 (English)",desc:"Best quality English"},{value:"F5",label:"F5 Base (English)",desc:"Standard English model"},{value:"F5-DE",label:"F5 German",desc:"German language"},{value:"F5-FR",label:"F5 French",desc:"French language"},{value:"F5-ES",label:"F5 Spanish",desc:"Spanish language"},{value:"F5-IT",label:"F5 Italian",desc:"Italian language"},{value:"F5-JP",label:"F5 Japanese",desc:"Japanese language"},{value:"E2",label:"E2-TTS",desc:"Alternative English model"}];function Ig({onOutput:e,onJobSubmitted:t}){const[n,a]=i.useState(null),[s,l]=i.useState(null),[o,c]=i.useState(null),[d,p]=i.useState(""),[v,g]=i.useState("F5v1"),[x,k]=i.useState(1),[w,z]=i.useState(!1),[F,f]=i.useState(0),u=i.useRef(null),h=i.useRef([]),y=i.useRef(null),j=i.useRef(null),I=i.useRef(null),[_,R]=i.useState(!1),[G,W]=i.useState(!1),[b,N]=i.useState(!1),[L,ee]=i.useState(!1),[T,ne]=i.useState(null),[ae,D]=i.useState(null),[U,q]=i.useState(null),V=i.useCallback(P=>{var te,K,de,pe;P.preventDefault();const O=((K=(te=P.dataTransfer)==null?void 0:te.files)==null?void 0:K[0])||((pe=(de=P.target)==null?void 0:de.files)==null?void 0:pe[0]);O&&Pg.some(Te=>O.type.includes(Te.split("/")[1]))?(a(O),l(URL.createObjectURL(O)),c(null),ne(null)):O&&ne("Please upload a valid audio file (WAV, MP3, FLAC, OGG)")},[]),H=async()=>{try{const P=await navigator.mediaDevices.getUserMedia({audio:!0}),O=new MediaRecorder(P,{mimeType:"audio/webm;codecs=opus"});h.current=[],u.current=O,O.ondataavailable=te=>{te.data.size>0&&h.current.push(te.data)},O.onstop=()=>{const te=new Blob(h.current,{type:"audio/webm"}),K=new File([te],"recording.webm",{type:"audio/webm"});a(K),l(URL.createObjectURL(te)),c(null),P.getTracks().forEach(de=>de.stop())},O.start(),z(!0),f(0),y.current=setInterval(()=>{f(te=>te+1)},1e3)}catch(P){ne("Failed to access microphone: "+P.message)}},Q=()=>{u.current&&w&&(u.current.stop(),z(!1),clearInterval(y.current))},C=async()=>{var O,te;if(!n)return null;const P=new FormData;P.append("file",n);try{const K=await We(`${oe}/upload`,P);if(K.ok&&((O=K.data)!=null&&O.path))return c(K.data.path),K.data.path;throw new Error(((te=K.data)==null?void 0:te.detail)||"Upload failed")}catch(K){throw new Error("Failed to upload voice sample: "+K.message)}},Y=async()=>{var P,O,te;if(!n||!d.trim()){ne("Please provide both a voice sample and text to speak");return}N(!0),ee(!0),ne(null),D(null),q(null);try{let K=o;K||(K=await C()),ee(!1);const de=await fa(`${oe}/voice-clone`,{voice_sample_path:K,text:d.trim(),model:v,speed:x});if(!de.ok)throw new Error(((P=de.data)==null?void 0:P.detail)||"Voice cloning request failed");(O=de.data)!=null&&O.prompt_id&&(D({promptId:de.data.prompt_id,model:((te=Uc.find(pe=>pe.value===v))==null?void 0:te.label)||v}),t&&t({prompt_id:de.data.prompt_id}))}catch(K){console.error("Voice cloning error:",K),ne(K.message)}finally{N(!1),ee(!1)}},M=()=>{a(null),l(null),c(null),D(null),j.current&&(j.current.pause(),j.current.currentTime=0),R(!1)},m=()=>{j.current&&(_?j.current.pause():j.current.play(),R(!_))},A=()=>{I.current&&(G?I.current.pause():I.current.play(),W(!G))},X=P=>{const O=Math.floor(P/60),te=P%60;return`${O}:${te.toString().padStart(2,"0")}`};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(vo,{size:18}),"Voice Sample (5-30 seconds recommended)"]}),n?r.jsxs("div",{className:"voice-preview",children:[r.jsxs("div",{className:"voice-file-info",children:[r.jsx(vo,{size:24}),r.jsxs("div",{className:"file-details",children:[r.jsx("span",{className:"filename",children:n.name}),r.jsxs("span",{className:"filesize",children:[(n.size/1024).toFixed(1)," KB"]})]}),r.jsxs("div",{className:"voice-controls",children:[r.jsx("button",{className:"icon-btn",onClick:m,children:_?r.jsx(yo,{size:18}):r.jsx(ua,{size:18})}),r.jsx("button",{className:"icon-btn danger",onClick:M,children:r.jsx(Cs,{size:18})})]})]}),r.jsx("audio",{ref:j,src:s,onEnded:()=>R(!1)}),o&&r.jsxs("div",{className:"upload-status",children:[r.jsx(Ns,{size:14})," Uploaded"]})]}):r.jsxs("div",{className:"voice-input-options",children:[r.jsxs("div",{className:"drop-zone",onDrop:V,onDragOver:P=>P.preventDefault(),onClick:()=>document.getElementById("voice-file-input").click(),children:[r.jsx(Ye,{size:32}),r.jsx("p",{children:"Drop audio file here or click to browse"}),r.jsx("span",{className:"supported-formats",children:"WAV, MP3, FLAC, OGG"}),r.jsx("input",{id:"voice-file-input",type:"file",accept:"audio/*",onChange:V,style:{display:"none"}})]}),r.jsx("div",{className:"divider-text",children:"or"}),r.jsx("button",{className:`record-btn ${w?"recording":""}`,onClick:w?Q:H,children:w?r.jsxs(r.Fragment,{children:[r.jsx("div",{className:"recording-indicator"}),r.jsxs("span",{children:["Stop Recording (",X(F),")"]})]}):r.jsxs(r.Fragment,{children:[r.jsx(gi,{size:20}),r.jsx("span",{children:"Record Voice Sample"})]})})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Text to Speak"}),r.jsx("textarea",{value:d,onChange:P=>p(P.target.value),placeholder:"Enter the text you want the cloned voice to speak...",rows:4,className:"prompt-textarea"}),r.jsxs("div",{className:"char-count",children:[d.length," characters"]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Model"}),r.jsx("div",{className:"model-grid",children:Uc.map(P=>r.jsxs("button",{className:`model-btn ${v===P.value?"active":""}`,onClick:()=>g(P.value),children:[r.jsx("span",{className:"model-name",children:P.label}),r.jsx("span",{className:"model-desc",children:P.desc})]},P.value))})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Speed"}),r.jsxs("div",{className:"slider-row",children:[r.jsx("input",{type:"range",min:.5,max:2,step:.1,value:x,onChange:P=>k(parseFloat(P.target.value))}),r.jsxs("span",{className:"slider-value",children:[x.toFixed(1),"x"]})]}),r.jsxs("div",{className:"slider-hints",children:[r.jsx("span",{children:">1.0 = slower"}),r.jsx("span",{children:"<1.0 = faster"})]})]}),ae&&r.jsxs("div",{className:"queued-notice",children:["✅ Job queued! Check the Queue panel for progress.",r.jsx("span",{className:"queued-mode",children:ae.model})]}),r.jsx("div",{className:"tool-section",children:r.jsx("button",{className:"generate-btn",onClick:Y,disabled:b||!n||!d.trim(),children:b?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:20,className:"spin"}),r.jsx("span",{children:L?"Uploading...":"Queueing..."})]}):r.jsxs(r.Fragment,{children:[r.jsx(hn,{size:20}),r.jsx("span",{children:"Clone Voice"})]})})}),T&&r.jsxs("div",{className:"error-message",children:[r.jsx(Qe,{size:16}),T]}),U&&r.jsxs("div",{className:"tool-section result-section",children:[r.jsxs("h3",{children:[r.jsx(hn,{size:18}),"Cloned Voice Result"]}),r.jsxs("div",{className:"audio-result",children:[r.jsx("audio",{ref:I,src:U.url,onEnded:()=>W(!1)}),r.jsxs("div",{className:"audio-controls",children:[r.jsx("button",{className:"play-btn",onClick:A,children:G?r.jsx(yo,{size:24}):r.jsx(ua,{size:24})}),r.jsx("span",{className:"filename",children:U.filename}),r.jsx("a",{href:U.url,download:U.filename,className:"download-btn",children:r.jsx(vt,{size:18})})]})]})]}),r.jsx("style",{children:`
        .voice-input-options {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        
        .drop-zone {
          border: 2px dashed #4a4a4a;
          border-radius: 12px;
          padding: 32px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .drop-zone:hover {
          border-color: #fbbf24;
          background: rgba(251, 191, 36, 0.05);
        }
        
        .drop-zone p {
          margin: 12px 0 4px;
          color: #ccc;
        }
        
        .supported-formats {
          font-size: 12px;
          color: #888;
        }
        
        .divider-text {
          text-align: center;
          color: #666;
          font-size: 13px;
        }
        
        .record-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 16px;
          border-radius: 12px;
          background: #2a2a2a;
          border: 2px solid #3a3a3a;
          color: #fff;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .record-btn:hover {
          border-color: #ef4444;
          background: rgba(239, 68, 68, 0.1);
        }
        
        .record-btn.recording {
          border-color: #ef4444;
          background: rgba(239, 68, 68, 0.2);
        }
        
        .recording-indicator {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          background: #ef4444;
          animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        
        .voice-preview {
          background: #1a1a1a;
          border-radius: 12px;
          padding: 16px;
        }
        
        .voice-file-info {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .file-details {
          flex: 1;
          display: flex;
          flex-direction: column;
        }
        
        .filename {
          color: #fff;
          font-size: 14px;
        }
        
        .filesize {
          color: #888;
          font-size: 12px;
        }
        
        .voice-controls {
          display: flex;
          gap: 8px;
        }
        
        .icon-btn {
          padding: 8px;
          border-radius: 8px;
          background: #2a2a2a;
          border: none;
          color: #fff;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .icon-btn:hover {
          background: #3a3a3a;
        }
        
        .icon-btn.danger:hover {
          background: rgba(239, 68, 68, 0.2);
          color: #ef4444;
        }
        
        .upload-status {
          display: flex;
          align-items: center;
          gap: 6px;
          margin-top: 8px;
          color: #22c55e;
          font-size: 12px;
        }
        
        .char-count {
          text-align: right;
          font-size: 12px;
          color: #666;
          margin-top: 4px;
        }
        
        .model-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
          gap: 8px;
        }
        
        .model-btn {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          padding: 12px;
          border-radius: 8px;
          background: #1a1a1a;
          border: 2px solid #2a2a2a;
          color: #fff;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .model-btn:hover {
          border-color: #4a4a4a;
        }
        
        .model-btn.active {
          border-color: #fbbf24;
          background: rgba(251, 191, 36, 0.1);
        }
        
        .model-name {
          font-size: 13px;
          font-weight: 500;
        }
        
        .model-desc {
          font-size: 11px;
          color: #888;
          margin-top: 2px;
        }
        
        .slider-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .slider-row input[type="range"] {
          flex: 1;
        }
        
        .slider-value {
          min-width: 50px;
          text-align: right;
          color: #fbbf24;
          font-weight: 500;
        }
        
        .slider-hints {
          display: flex;
          justify-content: space-between;
          font-size: 11px;
          color: #666;
          margin-top: 4px;
        }
        
        .result-section {
          background: rgba(34, 197, 94, 0.1);
          border: 1px solid rgba(34, 197, 94, 0.3);
          border-radius: 12px;
          padding: 16px;
        }
        
        .audio-result {
          margin-top: 12px;
        }
        
        .audio-controls {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .play-btn {
          width: 48px;
          height: 48px;
          border-radius: 50%;
          background: #fbbf24;
          border: none;
          color: #000;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s;
        }
        
        .play-btn:hover {
          background: #f59e0b;
          transform: scale(1.05);
        }
        
        .download-btn {
          margin-left: auto;
          padding: 8px 16px;
          border-radius: 8px;
          background: #2a2a2a;
          color: #fff;
          text-decoration: none;
          display: flex;
          align-items: center;
          gap: 6px;
          transition: all 0.2s;
        }
        
        .download-btn:hover {
          background: #3a3a3a;
        }
        
        .error-message {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 16px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          font-size: 13px;
        }
        
        .progress-bar {
          height: 4px;
          background: #2a2a2a;
          border-radius: 2px;
          margin-top: 12px;
          overflow: hidden;
        }
        
        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #fbbf24, #f59e0b);
          transition: width 0.3s;
        }
        
        .spin {
          animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `})]})}const Rg=["video/mp4","video/webm","video/quicktime"],Mg=["audio/wav","audio/mp3","audio/mpeg","audio/flac","audio/ogg","audio/webm"];function Fg({onOutput:e,onJobSubmitted:t}){const[n,a]=i.useState(null),[s,l]=i.useState(null),[o,c]=i.useState(null),[d,p]=i.useState(null),[v,g]=i.useState(null),[x,k]=i.useState(null),[w,z]=i.useState(1.5),[F,f]=i.useState(20),[u,h]=i.useState(-1),y=i.useRef(null),j=i.useRef(null),I=i.useRef(null),[_,R]=i.useState(!1),[G,W]=i.useState(!1),[b,N]=i.useState(null),[L,ee]=i.useState(null),[T,ne]=i.useState(null),ae=i.useCallback(Q=>{var Y,M,m,A;Q.preventDefault();const C=((M=(Y=Q.dataTransfer)==null?void 0:Y.files)==null?void 0:M[0])||((A=(m=Q.target)==null?void 0:m.files)==null?void 0:A[0]);C&&Rg.some(X=>C.type.includes(X.split("/")[1]))?(a(C),l(URL.createObjectURL(C)),c(null),N(null),ee(null)):C&&N("Please upload a valid video file (MP4, WebM)")},[]),D=i.useCallback(Q=>{var Y,M,m,A;Q.preventDefault();const C=((M=(Y=Q.dataTransfer)==null?void 0:Y.files)==null?void 0:M[0])||((A=(m=Q.target)==null?void 0:m.files)==null?void 0:A[0]);C&&Mg.some(X=>C.type.includes(X.split("/")[1]))?(p(C),g(URL.createObjectURL(C)),k(null),N(null),ee(null)):C&&N("Please upload a valid audio file (WAV, MP3, FLAC)")},[]),U=async Q=>{var Y,M;const C=new FormData;C.append("file",Q);try{const m=await We(`${oe}/upload`,C);if(m.ok&&((Y=m.data)!=null&&Y.path))return m.data.path;throw new Error(((M=m.data)==null?void 0:M.detail)||"Upload failed")}catch(m){throw new Error("Failed to upload file: "+m.message)}},q=async()=>{var Q,C;if(!n||!d){N("Please provide both a video and audio file");return}R(!0),W(!0),N(null),ee(null),ne(null);try{let Y=o;Y||(Y=await U(n),c(Y));let M=x;M||(M=await U(d),k(M)),W(!1);const m=await fa(`${oe}/lip-sync`,{video_path:Y,audio_path:M,lips_expression:w,inference_steps:F,seed:u===-1?Math.floor(Math.random()*2147483647):u});if(!m.ok)throw new Error(((Q=m.data)==null?void 0:Q.detail)||"Lip sync request failed");(C=m.data)!=null&&C.prompt_id&&(ee({promptId:m.data.prompt_id}),t&&t({prompt_id:m.data.prompt_id}))}catch(Y){console.error("Lip sync error:",Y),N(Y.message)}finally{R(!1),W(!1)}},V=()=>{a(null),l(null),c(null)},H=()=>{p(null),g(null),k(null)};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(Oh,{size:18}),"Input Video (with face)"]}),n?r.jsxs("div",{className:"media-preview",children:[r.jsx("video",{ref:y,src:s,controls:!0,className:"preview-video"}),r.jsxs("div",{className:"file-info-row",children:[r.jsx("span",{className:"filename",children:n.name}),r.jsx("button",{className:"icon-btn danger",onClick:V,children:r.jsx(Cs,{size:18})})]})]}):r.jsxs("div",{className:"drop-zone",onDrop:ae,onDragOver:Q=>Q.preventDefault(),onClick:()=>document.getElementById("video-file-input").click(),children:[r.jsx(Ye,{size:32}),r.jsx("p",{children:"Drop video file here or click to browse"}),r.jsx("span",{className:"supported-formats",children:"MP4, WebM"}),r.jsx("input",{id:"video-file-input",type:"file",accept:"video/*",onChange:ae,style:{display:"none"}})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(vo,{size:18}),"Audio Track (speech/dialogue)"]}),d?r.jsxs("div",{className:"audio-preview",children:[r.jsx("audio",{ref:j,src:v,controls:!0,className:"preview-audio"}),r.jsxs("div",{className:"file-info-row",children:[r.jsx("span",{className:"filename",children:d.name}),r.jsx("button",{className:"icon-btn danger",onClick:H,children:r.jsx(Cs,{size:18})})]})]}):r.jsxs("div",{className:"drop-zone",onDrop:D,onDragOver:Q=>Q.preventDefault(),onClick:()=>document.getElementById("audio-file-input").click(),children:[r.jsx(Ye,{size:32}),r.jsx("p",{children:"Drop audio file here or click to browse"}),r.jsx("span",{className:"supported-formats",children:"WAV, MP3, FLAC, OGG"}),r.jsx("input",{id:"audio-file-input",type:"file",accept:"audio/*",onChange:D,style:{display:"none"}})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(pa,{size:18}),"Settings"]}),r.jsxs("div",{className:"setting-row",children:[r.jsx("label",{children:"Lips Expression"}),r.jsxs("div",{className:"slider-row",children:[r.jsx("input",{type:"range",min:1,max:3,step:.1,value:w,onChange:Q=>z(parseFloat(Q.target.value))}),r.jsx("span",{className:"slider-value",children:w.toFixed(1)})]}),r.jsx("span",{className:"setting-hint",children:"Higher = more exaggerated lip movements"})]}),r.jsxs("div",{className:"setting-row",children:[r.jsx("label",{children:"Inference Steps"}),r.jsxs("div",{className:"slider-row",children:[r.jsx("input",{type:"range",min:10,max:50,step:5,value:F,onChange:Q=>f(parseInt(Q.target.value))}),r.jsx("span",{className:"slider-value",children:F})]}),r.jsx("span",{className:"setting-hint",children:"More steps = better quality, slower"})]}),r.jsxs("div",{className:"setting-row",children:[r.jsx("label",{children:"Seed"}),r.jsx("input",{type:"number",value:u,onChange:Q=>h(parseInt(Q.target.value)||-1),placeholder:"-1 for random",className:"seed-input"})]})]}),L&&r.jsx("div",{className:"queued-notice",children:"✅ Job queued! Check the Queue panel for progress."}),r.jsx("div",{className:"tool-section",children:r.jsx("button",{className:"generate-btn",onClick:q,disabled:_||!n||!d,children:_?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:20,className:"spin"}),r.jsx("span",{children:G?"Uploading...":"Queueing..."})]}):r.jsxs(r.Fragment,{children:[r.jsx(yr,{size:20}),r.jsx("span",{children:"Sync Lips"})]})})}),b&&r.jsxs("div",{className:"error-message",children:[r.jsx(Qe,{size:16}),b]}),T&&r.jsxs("div",{className:"tool-section result-section",children:[r.jsxs("h3",{children:[r.jsx(yr,{size:18}),"Lip Synced Result"]}),r.jsxs("div",{className:"video-result",children:[r.jsx("video",{ref:I,src:T.url,controls:!0,className:"result-video"}),r.jsxs("div",{className:"result-actions",children:[r.jsx("span",{className:"filename",children:T.filename}),r.jsxs("a",{href:T.url,download:T.filename,className:"download-btn",children:[r.jsx(vt,{size:18}),"Download"]})]})]})]}),r.jsx("style",{children:`
        .drop-zone {
          border: 2px dashed #4a4a4a;
          border-radius: 12px;
          padding: 32px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .drop-zone:hover {
          border-color: #fbbf24;
          background: rgba(251, 191, 36, 0.05);
        }
        
        .drop-zone p {
          margin: 12px 0 4px;
          color: #ccc;
        }
        
        .supported-formats {
          font-size: 12px;
          color: #888;
        }
        
        .media-preview, .audio-preview {
          background: #1a1a1a;
          border-radius: 12px;
          padding: 16px;
        }
        
        .preview-video, .result-video {
          width: 100%;
          max-height: 300px;
          border-radius: 8px;
          background: #000;
        }
        
        .preview-audio {
          width: 100%;
        }
        
        .file-info-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-top: 12px;
        }
        
        .filename {
          color: #ccc;
          font-size: 13px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        
        .icon-btn {
          padding: 8px;
          border-radius: 8px;
          background: #2a2a2a;
          border: none;
          color: #fff;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .icon-btn:hover {
          background: #3a3a3a;
        }
        
        .icon-btn.danger:hover {
          background: rgba(239, 68, 68, 0.2);
          color: #ef4444;
        }
        
        .setting-row {
          margin-bottom: 16px;
        }
        
        .setting-row label {
          display: block;
          margin-bottom: 8px;
          color: #ccc;
          font-size: 13px;
        }
        
        .slider-row {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .slider-row input[type="range"] {
          flex: 1;
        }
        
        .slider-value {
          min-width: 50px;
          text-align: right;
          color: #fbbf24;
          font-weight: 500;
        }
        
        .setting-hint {
          display: block;
          font-size: 11px;
          color: #666;
          margin-top: 4px;
        }
        
        .seed-input {
          width: 100%;
          padding: 10px 12px;
          border-radius: 8px;
          background: #1a1a1a;
          border: 1px solid #2a2a2a;
          color: #fff;
          font-size: 14px;
        }
        
        .result-section {
          background: rgba(34, 197, 94, 0.1);
          border: 1px solid rgba(34, 197, 94, 0.3);
          border-radius: 12px;
          padding: 16px;
        }
        
        .video-result {
          margin-top: 12px;
        }
        
        .result-actions {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-top: 12px;
        }
        
        .download-btn {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 10px 16px;
          border-radius: 8px;
          background: #fbbf24;
          color: #000;
          text-decoration: none;
          font-weight: 500;
          transition: all 0.2s;
        }
        
        .download-btn:hover {
          background: #f59e0b;
        }
        
        .error-message {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 12px 16px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          font-size: 13px;
        }
        
        .progress-bar {
          height: 4px;
          background: #2a2a2a;
          border-radius: 2px;
          margin-top: 12px;
          overflow: hidden;
        }
        
        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #fbbf24, #f59e0b);
          transition: width 0.3s;
        }
        
        .spin {
          animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `})]})}const Vc=[{id:"1:1",label:"1:1 (Square)",width:1024,height:1024},{id:"16:9",label:"16:9 (Widescreen)",width:1280,height:720},{id:"9:16",label:"9:16 (Portrait)",width:720,height:1280},{id:"4:3",label:"4:3 (Standard)",width:1024,height:768},{id:"3:4",label:"3:4 (Portrait)",width:768,height:1024},{id:"21:9",label:"21:9 (Ultrawide)",width:1344,height:576},{id:"3:2",label:"3:2 (Photo)",width:1152,height:768},{id:"2:3",label:"2:3 (Photo Portrait)",width:768,height:1152}],Lg=[{id:"center",label:"Center",icon:"⊕"},{id:"top",label:"Top",icon:"⬆️"},{id:"bottom",label:"Bottom",icon:"⬇️"},{id:"left",label:"Left",icon:"⬅️"},{id:"right",label:"Right",icon:"➡️"},{id:"top-left",label:"Top Left",icon:"↖️"},{id:"top-right",label:"Top Right",icon:"↗️"},{id:"bottom-left",label:"Bottom Left",icon:"↙️"},{id:"bottom-right",label:"Bottom Right",icon:"↘️"}],Bc=[{id:"sdxl",label:"SDXL (Quality)",file:"CyberRealisticPony_v8.safetensors"},{id:"flux",label:"Flux (Fast)",file:"flux1-dev-bnb-nf4.safetensors"}];function Dg({onJobSubmitted:e}){const[t,n]=i.useState(null),[a,s]=i.useState(null),[l,o]=i.useState({width:0,height:0}),[c,d]=i.useState(Vc[0]),[p,v]=i.useState("center"),[g,x]=i.useState(Bc[0]),[k,w]=i.useState(""),[z,F]=i.useState(25),[f,u]=i.useState(7),[h,y]=i.useState(.85),[j,I]=i.useState(32),[_,R]=i.useState(!1),[G,W]=i.useState(null),[b,N]=i.useState(null),[L,ee]=i.useState(!1),[T,ne]=i.useState(null),ae=i.useRef(null),D=i.useCallback(C=>{var M,m,A,X;C.preventDefault();const Y=((m=(M=C.dataTransfer)==null?void 0:M.files)==null?void 0:m[0])||((X=(A=C.target)==null?void 0:A.files)==null?void 0:X[0]);if(Y&&Y.type.startsWith("image/")){n(Y),W(null),N(null),ne(null);const P=URL.createObjectURL(Y),O=new Image;O.onload=()=>{o({width:O.naturalWidth,height:O.naturalHeight}),s(P)},O.src=P}},[]),U=C=>C.preventDefault(),q=async()=>{var C,Y,M;if(!t){N("Please upload an image first");return}R(!0),N(null),W(null),ne(null);try{const m=new FormData;m.append("image",t),m.append("target_width",c.width),m.append("target_height",c.height),m.append("position",p),m.append("prompt",k||"seamless natural extension, high quality"),m.append("model",g.file),m.append("steps",z),m.append("cfg",f),m.append("denoise",h),m.append("feathering",j);const A=await We(`${oe}/reframe`,m);if(!A.ok)throw new Error(((C=A.data)==null?void 0:C.detail)||"Reframe request failed");(Y=A.data)!=null&&Y.prompt_id?(ne({promptId:A.data.prompt_id,aspectRatio:c.label}),e&&e({prompt_id:A.data.prompt_id})):(M=A.data)!=null&&M.url&&W({url:A.data.url})}catch(m){console.error("❌ Reframe error:",m),N(m.message)}finally{R(!1)}},V=()=>{if(!(G!=null&&G.url))return;const C=document.createElement("a");C.href=G.url,C.download=`reframed_${c.id.replace(":","x")}_${Date.now()}.png`,C.click()},Q=(()=>{if(!l.width||!l.height)return null;const C=c.width,Y=c.height,M=l.width,m=l.height,A=C/M,X=Y/m,P=Math.min(A,X),O=Math.round(M*P),te=Math.round(m*P);let K=0,de=0;return p.includes("left")?K=0:p.includes("right")?K=C-O:K=(C-O)/2,p.includes("top")?de=0:p.includes("bottom")?de=Y-te:de=(Y-te)/2,{scaledW:O,scaledH:te,offsetX:K,offsetY:de,targetW:C,targetH:Y}})();return r.jsxs("div",{className:"space-y-4",children:[r.jsxs("div",{onClick:()=>{var C;return(C=ae.current)==null?void 0:C.click()},onDrop:D,onDragOver:U,className:"border-2 border-dashed border-gray-600 rounded-lg p-6 text-center cursor-pointer hover:border-purple-500 transition-colors",children:[r.jsx("input",{ref:ae,type:"file",accept:"image/*",onChange:D,className:"hidden"}),a?r.jsxs("div",{className:"flex flex-col items-center gap-2",children:[r.jsx("img",{src:a,alt:"Preview",className:"max-h-32 rounded"}),r.jsxs("span",{className:"text-sm text-gray-400",children:["Original: ",l.width,"×",l.height]}),r.jsx("span",{className:"text-xs text-gray-500",children:"Click to change"})]}):r.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[r.jsx(Ye,{className:"w-8 h-8"}),r.jsx("span",{children:"Drop image here or click to upload"})]})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Target Aspect Ratio"}),r.jsx("div",{className:"grid grid-cols-4 gap-2",children:Vc.map(C=>r.jsx("button",{onClick:()=>d(C),className:`px-3 py-2 text-sm rounded transition-colors ${c.id===C.id?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:C.label},C.id))}),r.jsxs("span",{className:"text-xs text-gray-500 mt-1 block",children:["Output: ",c.width,"×",c.height]})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Image Position"}),r.jsx("div",{className:"grid grid-cols-3 gap-2 w-40 mx-auto",children:["top-left","top","top-right","left","center","right","bottom-left","bottom","bottom-right"].map(C=>{var Y;return r.jsx("button",{onClick:()=>v(C),className:`p-2 text-lg rounded transition-colors ${p===C?"bg-purple-600":"bg-gray-700 hover:bg-gray-600"}`,title:C,children:((Y=Lg.find(M=>M.id===C))==null?void 0:Y.icon)||"○"},C)})})]}),Q&&r.jsxs("div",{className:"bg-gray-800 rounded-lg p-4",children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Layout Preview"}),r.jsxs("div",{className:"relative mx-auto border border-gray-600 bg-gray-900",style:{width:Math.min(300,Q.targetW/3),height:Math.min(300,Q.targetH/3),aspectRatio:`${Q.targetW} / ${Q.targetH}`},children:[r.jsx("div",{className:"absolute inset-0 bg-stripes opacity-30"}),r.jsx("div",{className:"absolute bg-purple-600/50 border-2 border-purple-400 flex items-center justify-center text-xs",style:{width:`${Q.scaledW/Q.targetW*100}%`,height:`${Q.scaledH/Q.targetH*100}%`,left:`${Q.offsetX/Q.targetW*100}%`,top:`${Q.offsetY/Q.targetH*100}%`},children:"Original"})]}),r.jsx("p",{className:"text-xs text-gray-500 text-center mt-2",children:"Purple = original image, striped = AI-generated fill"})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Fill Prompt (optional)"}),r.jsx("textarea",{value:k,onChange:C=>w(C.target.value),placeholder:"Describe what should appear in the extended areas...",className:"w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 resize-none",rows:2})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Model"}),r.jsx("div",{className:"flex gap-2",children:Bc.map(C=>r.jsx("button",{onClick:()=>x(C),className:`flex-1 px-3 py-2 text-sm rounded transition-colors ${g.id===C.id?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:C.label},C.id))})]}),r.jsxs("div",{className:"border border-gray-700 rounded-lg overflow-hidden",children:[r.jsxs("button",{onClick:()=>ee(!L),className:"w-full px-4 py-2 bg-gray-800 flex items-center justify-between text-gray-300 hover:bg-gray-750",children:[r.jsx("span",{className:"text-sm font-medium",children:"Advanced Settings"}),r.jsx(Tt,{className:`w-4 h-4 transition-transform ${L?"rotate-180":""}`})]}),L&&r.jsxs("div",{className:"p-4 space-y-4 bg-gray-850",children:[r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Steps: ",z]}),r.jsx("input",{type:"range",min:10,max:50,value:z,onChange:C=>F(Number(C.target.value)),className:"w-full accent-purple-500"})]}),r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["CFG Scale: ",f]}),r.jsx("input",{type:"range",min:1,max:15,step:.5,value:f,onChange:C=>u(Number(C.target.value)),className:"w-full accent-purple-500"})]}),r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Denoise: ",h.toFixed(2)]}),r.jsx("input",{type:"range",min:.5,max:1,step:.05,value:h,onChange:C=>y(Number(C.target.value)),className:"w-full accent-purple-500"}),r.jsx("span",{className:"text-xs text-gray-500",children:"Higher = more creative fill"})]}),r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Edge Feathering: ",j,"px"]}),r.jsx("input",{type:"range",min:0,max:64,step:8,value:j,onChange:C=>I(Number(C.target.value)),className:"w-full accent-purple-500"}),r.jsx("span",{className:"text-xs text-gray-500",children:"Blend between original and fill"})]})]})]}),r.jsx("button",{onClick:q,disabled:_||!t,className:"w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors",children:_?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{className:"w-5 h-5 animate-spin"}),"Queueing..."]}):r.jsxs(r.Fragment,{children:[r.jsx(Wh,{className:"w-5 h-5"}),"Reframe Image"]})}),T&&r.jsxs("div",{className:"p-3 bg-green-900/50 border border-green-700 rounded-lg text-green-200 text-sm",children:["✅ Reframe job queued! (",T.aspectRatio,") - Check queue panel for progress"]}),b&&r.jsx("div",{className:"p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm",children:b}),G&&r.jsxs("div",{className:"space-y-3",children:[r.jsx("div",{className:"rounded-lg overflow-hidden border border-gray-700",children:r.jsx("img",{src:G.url,alt:"Reframed",className:"w-full"})}),r.jsxs("div",{className:"flex gap-2",children:[r.jsxs("button",{onClick:V,className:"flex-1 py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center justify-center gap-2",children:[r.jsx(vt,{className:"w-4 h-4"}),"Download"]}),r.jsxs("button",{onClick:()=>{n(null),s(null),W(null),fetch(G.url).then(C=>C.blob()).then(C=>{const Y=new File([C],"reframed.png",{type:"image/png"});n(Y),s(G.url);const M=new Image;M.onload=()=>o({width:M.naturalWidth,height:M.naturalHeight}),M.src=G.url})},className:"flex-1 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center justify-center gap-2",children:[r.jsx(ix,{className:"w-4 h-4"}),"Use as Input"]})]})]}),r.jsxs("div",{className:"text-xs text-gray-500 space-y-1",children:[r.jsxs("p",{children:["💡 ",r.jsx("strong",{children:"Reframe"})," extends your image to a new aspect ratio using AI outpainting."]}),r.jsx("p",{children:"📐 The original image will be placed according to the position you select."}),r.jsx("p",{children:"🎨 Use the prompt to guide what should appear in the extended areas."})]})]})}const Wc=[{id:"inswapper",label:"InSwapper (Best Quality)",description:"High quality, slower"},{id:"simswap",label:"SimSwap (Fast)",description:"Faster, good quality"}],Og=[{id:"none",label:"None"},{id:"gfpgan",label:"GFPGAN (Faces)"},{id:"codeformer",label:"CodeFormer (Natural)"},{id:"both",label:"Both (Best)"}];function Ag({onJobSubmitted:e}){const[t,n]=i.useState(null),[a,s]=i.useState(null),[l,o]=i.useState(null),[c,d]=i.useState(null),[p,v]=i.useState(Wc[0]),[g,x]=i.useState("gfpgan"),[k,w]=i.useState(1),[z,F]=i.useState(.8),[f,u]=i.useState(0),[h,y]=i.useState(!1),[j,I]=i.useState(!1),[_,R]=i.useState(null),[G,W]=i.useState(null),[b,N]=i.useState(null),[L,ee]=i.useState(!1),[T,ne]=i.useState(null),ae=i.useRef(null),D=i.useRef(null),U=i.useCallback(M=>{var A,X,P,O;M.preventDefault();const m=((X=(A=M.dataTransfer)==null?void 0:A.files)==null?void 0:X[0])||((O=(P=M.target)==null?void 0:P.files)==null?void 0:O[0]);if(m&&(m.type.startsWith("image/")||m.type.startsWith("video/"))){n(m),R(null),W(null),N(null),ne(null);const te=URL.createObjectURL(m);s(te)}},[]),q=i.useCallback(M=>{var A,X,P,O;M.preventDefault();const m=((X=(A=M.dataTransfer)==null?void 0:A.files)==null?void 0:X[0])||((O=(P=M.target)==null?void 0:P.files)==null?void 0:O[0]);if(m&&m.type.startsWith("image/")){o(m),R(null),W(null),ne(null);const te=URL.createObjectURL(m);d(te)}},[]),V=M=>M.preventDefault(),H=async()=>{var M,m;if(t){I(!0),W(null);try{const A=new FormData;A.append("image",t);const X=await We(`${oe}/detect-faces`,A);if(X.ok&&((M=X.data)!=null&&M.faces))N(X.data.faces);else throw new Error(((m=X.data)==null?void 0:m.detail)||"Face detection failed")}catch(A){console.error("❌ Face detection error:",A),W(A.message)}finally{I(!1)}}},Q=async()=>{var M,m,A;if(!t||!l){W("Please upload both target and source face images");return}I(!0),W(null),R(null),ne(null);try{const X=new FormData;X.append("target",t),X.append("source",l),X.append("model",p.id),X.append("enhance",g),X.append("strength",k),X.append("blend",z),X.append("face_index",h?-1:f);const P=await We(`${oe}/face-swap`,X);if(!P.ok)throw new Error(((M=P.data)==null?void 0:M.detail)||"Face swap request failed");(m=P.data)!=null&&m.prompt_id?(ne({promptId:P.data.prompt_id,model:p.label}),e&&e({prompt_id:P.data.prompt_id})):(A=P.data)!=null&&A.url&&R({url:P.data.url})}catch(X){console.error("❌ FaceSwap error:",X),W(X.message)}finally{I(!1)}},C=()=>{if(!(_!=null&&_.url))return;const M=t!=null&&t.type.startsWith("video/")?"mp4":"png",m=document.createElement("a");m.href=_.url,m.download=`face_swap_${Date.now()}.${M}`,m.click()},Y=()=>{const M=t,m=a;n(l),s(c),o(M),d(m),R(null),N(null),ne(null)};return r.jsxs("div",{className:"space-y-4",children:[r.jsxs("div",{className:"grid grid-cols-2 gap-4",children:[r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Target (face to replace)"}),r.jsxs("div",{onClick:()=>{var M;return(M=ae.current)==null?void 0:M.click()},onDrop:U,onDragOver:V,className:"border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-purple-500 transition-colors aspect-square flex items-center justify-center",children:[r.jsx("input",{ref:ae,type:"file",accept:"image/*,video/*",onChange:U,className:"hidden"}),a?r.jsxs("div",{className:"relative w-full h-full",children:[t!=null&&t.type.startsWith("video/")?r.jsx("video",{src:a,className:"w-full h-full object-cover rounded",muted:!0}):r.jsx("img",{src:a,alt:"Target",className:"w-full h-full object-cover rounded"}),b&&r.jsxs("div",{className:"absolute bottom-1 right-1 bg-black/70 px-2 py-1 rounded text-xs",children:[b.length," face",b.length!==1?"s":""," detected"]})]}):r.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[r.jsx(Ye,{className:"w-6 h-6"}),r.jsx("span",{className:"text-xs",children:"Target image/video"})]})]})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Source (face to use)"}),r.jsxs("div",{onClick:()=>{var M;return(M=D.current)==null?void 0:M.click()},onDrop:q,onDragOver:V,className:"border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-blue-500 transition-colors aspect-square flex items-center justify-center",children:[r.jsx("input",{ref:D,type:"file",accept:"image/*",onChange:q,className:"hidden"}),c?r.jsx("img",{src:c,alt:"Source",className:"w-full h-full object-cover rounded"}):r.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[r.jsx(bx,{className:"w-6 h-6"}),r.jsx("span",{className:"text-xs",children:"Source face"})]})]})]})]}),(t||l)&&r.jsxs("button",{onClick:Y,className:"w-full py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center justify-center gap-2 text-sm",children:[r.jsx(mn,{className:"w-4 h-4"}),"Swap Target ↔ Source"]}),t&&!t.type.startsWith("video/")&&r.jsxs("button",{onClick:H,disabled:j,className:"w-full py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center justify-center gap-2 text-sm",children:[r.jsx(jo,{className:"w-4 h-4"}),"Detect Faces"]}),b&&b.length>1&&r.jsxs("div",{className:"bg-gray-800 rounded-lg p-3 space-y-2",children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300",children:"Select Face to Replace"}),r.jsx("div",{className:"flex items-center gap-4",children:r.jsxs("label",{className:"flex items-center gap-2",children:[r.jsx("input",{type:"checkbox",checked:h,onChange:M=>y(M.target.checked),className:"rounded bg-gray-700 border-gray-600"}),r.jsx("span",{className:"text-sm text-gray-300",children:"Swap all faces"})]})}),!h&&r.jsx("div",{className:"flex gap-2 flex-wrap",children:b.map((M,m)=>r.jsxs("button",{onClick:()=>u(m),className:`px-3 py-1 text-sm rounded ${f===m?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:["Face ",m+1]},m))})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Model"}),r.jsx("div",{className:"space-y-2",children:Wc.map(M=>r.jsxs("button",{onClick:()=>v(M),className:`w-full px-3 py-2 text-left rounded transition-colors ${p.id===M.id?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:[r.jsx("div",{className:"font-medium text-sm",children:M.label}),r.jsx("div",{className:"text-xs opacity-70",children:M.description})]},M.id))})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Face Enhancement"}),r.jsx("div",{className:"grid grid-cols-2 gap-2",children:Og.map(M=>r.jsx("button",{onClick:()=>x(M.id),className:`px-3 py-2 text-sm rounded transition-colors ${g===M.id?"bg-blue-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:M.label},M.id))})]}),r.jsxs("div",{className:"border border-gray-700 rounded-lg overflow-hidden",children:[r.jsxs("button",{onClick:()=>ee(!L),className:"w-full px-4 py-2 bg-gray-800 flex items-center justify-between text-gray-300 hover:bg-gray-750",children:[r.jsx("span",{className:"text-sm font-medium",children:"Advanced Settings"}),r.jsx(Tt,{className:`w-4 h-4 transition-transform ${L?"rotate-180":""}`})]}),L&&r.jsxs("div",{className:"p-4 space-y-4 bg-gray-850",children:[r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Swap Strength: ",k.toFixed(2)]}),r.jsx("input",{type:"range",min:.1,max:1,step:.05,value:k,onChange:M=>w(Number(M.target.value)),className:"w-full accent-purple-500"}),r.jsx("span",{className:"text-xs text-gray-500",children:"Lower = more original features preserved"})]}),r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Edge Blend: ",z.toFixed(2)]}),r.jsx("input",{type:"range",min:0,max:1,step:.05,value:z,onChange:M=>F(Number(M.target.value)),className:"w-full accent-purple-500"}),r.jsx("span",{className:"text-xs text-gray-500",children:"Blend face edges with background"})]})]})]}),r.jsxs("div",{className:"flex items-start gap-2 p-3 bg-yellow-900/30 border border-yellow-700/50 rounded-lg",children:[r.jsx(mh,{className:"w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5"}),r.jsxs("div",{className:"text-sm text-yellow-200",children:[r.jsx("strong",{children:"Ethical Use:"})," Only use face swap with consent of all parties involved. Creating non-consensual deepfakes is illegal in many jurisdictions."]})]}),r.jsx("button",{onClick:Q,disabled:j||!t||!l,className:"w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors",children:j?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{className:"w-5 h-5 animate-spin"}),"Swapping... ",progress>0&&`${Math.round(progress)}%`]}):r.jsxs(r.Fragment,{children:[r.jsx(jo,{className:"w-5 h-5"}),"Swap Face"]})}),G&&r.jsx("div",{className:"p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm",children:G}),_&&r.jsxs("div",{className:"space-y-3",children:[r.jsx("div",{className:"rounded-lg overflow-hidden border border-gray-700",children:t!=null&&t.type.startsWith("video/")?r.jsx("video",{src:_.url,controls:!0,className:"w-full"}):r.jsx("img",{src:_.url,alt:"Result",className:"w-full"})}),r.jsxs("button",{onClick:C,className:"w-full py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center justify-center gap-2",children:[r.jsx(vt,{className:"w-4 h-4"}),"Download Result"]})]}),r.jsxs("div",{className:"text-xs text-gray-500 space-y-1",children:[r.jsxs("p",{children:["👤 ",r.jsx("strong",{children:"Face Swap"})," replaces faces in images or videos using AI."]}),r.jsx("p",{children:"📸 For best results, use clear frontal face photos with good lighting."}),r.jsx("p",{children:"🎬 Video processing may take longer depending on length and resolution."})]})]})}function $g({title:e}){return r.jsxs("div",{className:"tool-coming-soon",children:[r.jsx("div",{className:"tool-title",children:e}),r.jsx("div",{className:"muted",children:"Missing backend endpoint (planned for v2)."})]})}const Hc=e=>{if(!e||isNaN(e))return null;const t=Math.floor(e/60),n=Math.floor(e%60);return`${t}:${n.toString().padStart(2,"0")}`},jp="oelala_media_favorites",bp="oelala_media_profile",Va={"1280x1024":{cols:4,label:"1280×1024"},"1080p":{cols:5,label:"1080p"},"1440p":{cols:6,label:"1440p"},"4k":{cols:8,label:"4K"}},Gc=()=>{const e=window.innerWidth;return e<=1280?"1280x1024":e<=1920?"1080p":e<=2560?"1440p":"4k"},Ug=()=>{try{return localStorage.getItem(bp)||"auto"}catch{return"auto"}},kl=e=>{try{localStorage.setItem(bp,e)}catch(t){console.error("Failed to save profile:",t)}},Vg=()=>{try{const e=localStorage.getItem(jp);return e?new Set(JSON.parse(e)):new Set}catch{return new Set}},Bg=e=>{try{localStorage.setItem(jp,JSON.stringify([...e]))}catch(t){console.error("Failed to save favorites:",t)}};function Ur({filter:e="all",selectionMode:t=!1,onSelectItem:n=null}){var ba,kn,wa,Hs,ka,kr,Sn,Nn,wt,Gs,Ar,Sa,ft;const[a,s]=i.useState([]),[l,o]=i.useState(!1),[c,d]=i.useState(""),[p,v]=i.useState({videos:0,images:0,audio:0}),[g,x]=i.useState(null),[k,w]=i.useState(new Set),[z,F]=i.useState(null),[f,u]=i.useState(!1),[h,y]=i.useState(!1),[j,I]=i.useState(null),[_,R]=i.useState(Vg),[G,W]=i.useState("date"),[b,N]=i.useState("desc"),[L,ee]=i.useState("all"),[T,ne]=i.useState(""),[ae,D]=i.useState(!0),[U,q]=i.useState(Ug),V=U==="auto"?Gc():U,Q=(Va[V]||Va["1080p"]).cols,[C,Y]=i.useState(!1),[M,m]=i.useState(100),[A,X]=i.useState(320),[P,O]=i.useState({}),te=i.useRef(null);i.useEffect(()=>{const S=()=>{if(te.current){const ie=(te.current.clientWidth-32-12*(Q-1))/Q,ye=Math.round(ie*(16/9));X(ye)}};return S(),window.addEventListener("resize",S),()=>window.removeEventListener("resize",S)},[Q]),i.useEffect(()=>{m(100)},[L,G,b,a]);const K=S=>{const{scrollTop:$,clientHeight:Z,scrollHeight:ie}=S.target;ie-$-Z<1e3&&m(ye=>Math.min(ye+50,pe.length))},de=(S,$)=>{$==null||$.stopPropagation(),R(Z=>{const ie=new Set(Z);return ie.has(S)?ie.delete(S):ie.add(S),Bg(ie),ie})},pe=i.useMemo(()=>{let S=[...a];if(L==="favorites"?S=S.filter($=>_.has($.filename)):L==="non-favorites"&&(S=S.filter($=>!_.has($.filename))),T.trim()){const $=T.toLowerCase().trim();S=S.filter(Z=>{var ie,ye,ce,me,at,Zt;return!!(Z.filename.toLowerCase().includes($)||(ye=(ie=Z.metadata)==null?void 0:ie.positive_prompt)!=null&&ye.toLowerCase().includes($)||(me=(ce=Z.metadata)==null?void 0:ce.prompt)!=null&&me.toLowerCase().includes($)||(Zt=(at=Z.metadata)==null?void 0:at.negative_prompt)!=null&&Zt.toLowerCase().includes($))})}return S.sort(($,Z)=>{let ie=0;switch(G){case"name":ie=$.filename.localeCompare(Z.filename);break;case"size":ie=($.size||0)-(Z.size||0);break;case"favorites":const ye=_.has($.filename)?1:0,ce=_.has(Z.filename)?1:0;ie=ye-ce;break;case"non-favorites":const me=_.has($.filename)?0:1,at=_.has(Z.filename)?0:1;ie=me-at;break;case"date":default:ie=($.mtime||0)-(Z.mtime||0);break}return b==="desc"?-ie:ie}),S},[a,G,b,L,_,T]),Te=i.useCallback(async()=>{o(!0),d("");try{const $=await fetch(`${oe}/list-comfyui-media?type=${e==="prompts"?"all":e}&grouped=true&include_metadata=true&hide_start_images=${ae}`);if(!$.ok)throw new Error("Failed to fetch media");const Z=await $.json();let ie=Z.media||[];e==="prompts"&&(ie=ie.filter(ye=>{var ce,me;return((ce=ye.metadata)==null?void 0:ce.positive_prompt)||((me=ye.metadata)==null?void 0:me.prompt)})),s(ie),v({videos:Z.videos||0,images:Z.images||0,audio:Z.audio||0}),w(new Set)}catch(S){d(S.message)}finally{o(!1)}},[e,ae]);i.useEffect(()=>{Te()},[Te]),i.useEffect(()=>{const S=$=>{if($.key==="?"||$.key==="/"&&$.shiftKey){$.preventDefault(),Y(Z=>!Z);return}if($.key==="+"||$.key==="="){$.preventDefault();const Z=["auto","1280x1024","1080p","1440p","4k"];q(ie=>{const ye=Z.indexOf(ie),ce=Z[(ye+1)%Z.length];return kl(ce),ce});return}if($.key==="-"||$.key==="_"){$.preventDefault();const Z=["auto","1280x1024","1080p","1440p","4k"];q(ie=>{const ye=Z.indexOf(ie),ce=Z[(ye-1+Z.length)%Z.length];return kl(ce),ce});return}if(g!==null&&($.key==="Escape"&&(x(null),Y(!1)),$.key==="ArrowLeft"&&x(Z=>Z>0?Z-1:pe.length-1),$.key==="ArrowRight"&&x(Z=>Z<pe.length-1?Z+1:0),$.key==="f"||$.key==="F"||$.key==="h"||$.key==="H")){const Z=pe[g];Z&&de(Z.filename)}};return window.addEventListener("keydown",S),()=>{window.removeEventListener("keydown",S)}},[g,pe,_]);const nt=(S,$)=>{if($.target.closest(".select-checkbox")){$.stopPropagation(),Pt(S,$);return}if(t&&n){const Z=pe[S];n(Z);return}x(S)},Pt=(S,$)=>{$==null||$.stopPropagation(),w(Z=>{const ie=new Set(Z);if($!=null&&$.shiftKey&&z!==null){const ye=Math.min(z,S),ce=Math.max(z,S);for(let me=ye;me<=ce;me++)ie.add(me)}else $!=null&&$.ctrlKey||$!=null&&$.metaKey,ie.has(S)?ie.delete(S):ie.add(S);return ie}),F(S)},bt=()=>{w(new Set(a.map((S,$)=>$)))},ya=()=>{w(new Set)},yn=async()=>{if(k.size===0)return;const S=Array.from(k).map(ce=>{var me;return(me=pe[ce])==null?void 0:me.filename}).filter(Boolean);if(S.length===0){d("No valid items selected for deletion");return}const $=S.filter(ce=>_.has(ce)),Z=$.length;let ie=`Delete ${S.length} item${S.length>1?"s":""} and their associated files (source images, metadata)?`;if(Z>0&&(ie=`⚠️ WARNING: ${Z} favorite${Z>1?"s":""} selected!

${ie}

Favorites to delete:
• ${$.slice(0,5).join(`
• `)}${Z>5?`
• ... and ${Z-5} more`:""}`),!!window.confirm(ie)){u(!0);try{const ce=await fetch(`${oe}/delete-comfyui-media`,{method:"DELETE",headers:{"Content-Type":"application/json"},body:JSON.stringify({filenames:S})});if(!ce.ok)throw new Error("Failed to delete");const me=await ce.json();console.log("Deleted:",me),await Te()}catch(ce){d(`Delete failed: ${ce.message}`)}finally{u(!1)}}},jn=(S,$)=>{$==null||$.stopPropagation();const Z=document.createElement("a");Z.href=`${oe}${S.url}`,Z.download=S.filename,Z.click()},bn=async()=>{if(k.size===0)return;const S=pe.filter($=>k.has($.filename));for(let $=0;$<S.length;$++){const Z=S[$],ie=document.createElement("a");ie.href=`${oe}${Z.url}`,ie.download=Z.filename,ie.click(),$<S.length-1&&await new Promise(ye=>setTimeout(ye,300))}},ja=async(S,$)=>{$==null||$.stopPropagation();try{const Z=await fetch(`${oe}/comfyui-metadata/${S.filename}`);if(!Z.ok)throw new Error("No metadata available");const ie=await Z.json(),ye=new Blob([JSON.stringify(ie.metadata,null,2)],{type:"application/json"}),ce=URL.createObjectURL(ye),me=document.createElement("a");me.href=ce,me.download=`${S.base_name||S.filename.replace(/\.[^/.]+$/,"")}_metadata.json`,me.click(),URL.revokeObjectURL(ce)}catch(Z){console.error("Failed to download metadata:",Z)}},It=S=>S<1024?`${S} B`:S<1024*1024?`${(S/1024).toFixed(1)} KB`:`${(S/1024/1024).toFixed(1)} MB`,ve=g!==null?pe[g]:null,wn=a.filter(S=>_.has(S.filename)).length;return r.jsxs("div",{style:{display:"flex",flexDirection:"column",height:"100%",backgroundColor:"var(--bg-primary)"},children:[r.jsx("style",{children:`
        /* ========== MEDIA GRID ========== */
        .media-grid {
          display: grid;
          gap: 12px;
          padding: 16px;
        }

        /* ========== THUMBNAIL CARD ========== */
        .thumb-card {
          position: relative;
          width: 100%;
          border-radius: 8px;
          overflow: hidden;
          cursor: pointer;
          background: #111;
        }
        .thumb-card:hover {
          outline: 2px solid var(--accent-color, #a855f7);
          z-index: 10;
        }
        .thumb-card.selected {
          outline: 3px solid var(--accent-color, #a855f7);
        }
        .thumb-card video,
        .thumb-card img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          display: block;
        }
        
        /* ========== AUDIO THUMBNAIL ========== */
        .audio-thumb {
          width: 100%;
          height: 100%;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        .audio-thumb .audio-icon {
          font-size: 3rem;
          margin-bottom: 8px;
        }
        .audio-thumb audio {
          display: none;
        }
        .audio-lightbox {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 40px;
          background: rgba(0,0,0,0.8);
          border-radius: 12px;
        }
        .audio-lightbox .audio-icon-large {
          font-size: 6rem;
          margin-bottom: 20px;
        }
        .audio-lightbox .audio-filename {
          color: var(--text-primary);
          font-size: 1.2rem;
          margin-bottom: 10px;
        }

        /* ========== SELECTION CHECKBOX ========== */
        .select-checkbox {
          position: absolute;
          top: 8px;
          left: 8px;
          width: 24px;
          height: 24px;
          border-radius: 6px;
          background: rgba(0,0,0,0.7);
          border: 2px solid rgba(255,255,255,0.8);
          opacity: 0;
          transition: opacity 0.15s;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          z-index: 20;
        }
        .thumb-card:hover .select-checkbox,
        .thumb-card.selected .select-checkbox {
          opacity: 1;
        }
        .thumb-card.selected .select-checkbox {
          background: var(--accent-color, #a855f7);
          border-color: var(--accent-color, #a855f7);
        }

        /* ========== FAVORITE BUTTON ========== */
        .favorite-btn {
          position: absolute;
          top: 8px;
          left: 40px;
          width: 24px;
          height: 24px;
          border-radius: 6px;
          background: rgba(0,0,0,0.7);
          border: 2px solid rgba(255,255,255,0.8);
          opacity: 0;
          transition: opacity 0.15s;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          z-index: 20;
        }
        .thumb-card:hover .favorite-btn {
          opacity: 1;
        }
        .favorite-btn.is-favorite {
          opacity: 1;
          background: #ef4444;
          border-color: #ef4444;
        }

        /* ========== PROMPT BUBBLE BUTTON ========== */
        .prompt-bubble-btn {
          position: absolute;
          top: 6px;
          right: 6px;
          width: 24px;
          height: 24px;
          border-radius: 4px;
          background: transparent;
          border: none;
          opacity: 0;
          transition: all 0.15s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          z-index: 21;
          font-size: 16px;
          line-height: 1;
          padding: 0;
          text-shadow: 0 1px 3px rgba(0,0,0,0.8);
        }
        .thumb-card:hover .prompt-bubble-btn {
          opacity: 1;
        }
        .prompt-bubble-btn:hover {
          transform: scale(1.2);
        }

        /* ========== PROMPT POPUP ========== */
        .prompt-popup-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0,0,0,0.5);
          z-index: 1000;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .prompt-popup {
          background: var(--bg-secondary, #1f1f1f);
          border: 1px solid var(--border-color, #333);
          border-radius: 12px;
          padding: 20px;
          max-width: 600px;
          width: 90%;
          max-height: 80vh;
          overflow-y: auto;
          box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }
        .prompt-popup-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
          padding-bottom: 12px;
          border-bottom: 1px solid var(--border-color, #333);
        }
        .prompt-popup-title {
          font-size: 1rem;
          font-weight: 600;
          color: var(--text-primary, #fff);
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .prompt-popup-close {
          background: none;
          border: none;
          color: var(--text-muted, #888);
          cursor: pointer;
          padding: 4px;
          border-radius: 4px;
        }
        .prompt-popup-close:hover {
          background: rgba(255,255,255,0.1);
          color: var(--text-primary, #fff);
        }
        .prompt-popup-content {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }
        .prompt-section {
          background: var(--bg-tertiary, #2a2a2a);
          padding: 12px;
          border-radius: 8px;
        }
        .prompt-section-label {
          font-size: 0.75rem;
          font-weight: 600;
          color: var(--text-muted, #888);
          margin-bottom: 8px;
          display: flex;
          align-items: center;
          gap: 6px;
        }
        .prompt-section-text {
          font-size: 0.9rem;
          color: var(--text-primary, #fff);
          line-height: 1.5;
          white-space: pre-wrap;
          word-break: break-word;
        }
        .prompt-copy-btn {
          background: var(--accent-color, #a855f7);
          border: none;
          color: #fff;
          padding: 8px 16px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 0.85rem;
          display: flex;
          align-items: center;
          gap: 6px;
          margin-top: 12px;
        }
        .prompt-copy-btn:hover {
          opacity: 0.9;
        }
        .prompt-media-preview {
          width: 80px;
          height: 80px;
          object-fit: cover;
          border-radius: 8px;
        }

        /* ========== SOURCE IMAGE BADGE ========== */
        .source-image-badge {
          position: absolute;
          top: 8px;
          right: 40px;
          padding: 3px 6px;
          border-radius: 4px;
          background: rgba(59, 130, 246, 0.9);
          color: #fff;
          font-size: 0.6rem;
          display: flex;
          align-items: center;
          gap: 3px;
          z-index: 20;
        }


        /* ========== MEDIA OVERLAY (hover info) ========== */
        .media-overlay {
          position: absolute;
          bottom: 0;
          left: 0;
          right: 0;
          padding: 8px;
          background: linear-gradient(transparent, rgba(0,0,0,0.8));
          opacity: 0;
          transition: opacity 0.15s;
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
        }
        .thumb-card:hover .media-overlay {
          opacity: 1;
        }
        .media-filename {
          font-size: 0.7rem;
          color: #fff;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          max-width: 70%;
        }
        .media-size {
          font-size: 0.65rem;
          color: rgba(255,255,255,0.6);
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .media-duration {
          display: inline-flex;
          align-items: center;
          gap: 3px;
          background: rgba(0,0,0,0.4);
          padding: 1px 5px;
          border-radius: 3px;
        }
        .overlay-buttons {
          display: flex;
          gap: 4px;
        }
        .overlay-btn {
          padding: 4px;
          border-radius: 4px;
          background: rgba(255,255,255,0.2);
          border: none;
          color: #fff;
          cursor: pointer;
        }
        .overlay-btn:hover {
          background: rgba(255,255,255,0.3);
        }

        /* ========== LIGHTBOX ========== */
        .lightbox-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0,0,0,0.95);
          z-index: 1000;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .lightbox-content {
          max-width: 90vw;
          max-height: 85vh;
          position: relative;
        }
        .lightbox-content video,
        .lightbox-content img {
          max-width: 90vw;
          max-height: 85vh;
          border-radius: 8px;
        }
        .lightbox-nav {
          position: absolute;
          top: 50%;
          transform: translateY(-50%);
          width: 48px;
          height: 48px;
          border-radius: 50%;
          background: rgba(255,255,255,0.1);
          border: none;
          color: #fff;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .lightbox-nav:hover {
          background: rgba(255,255,255,0.2);
        }
        .lightbox-close {
          position: absolute;
          top: 20px;
          right: 20px;
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background: rgba(255,255,255,0.1);
          border: none;
          color: #fff;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1001;
        }
        .lightbox-close:hover {
          background: rgba(255,255,255,0.2);
        }
        .lightbox-info {
          position: absolute;
          bottom: 20px;
          left: 50%;
          transform: translateX(-50%);
          background: rgba(0,0,0,0.7);
          padding: 12px 20px;
          border-radius: 8px;
          display: flex;
          gap: 16px;
          align-items: center;
        }
        .lightbox-metadata {
          position: absolute;
          top: 20px;
          left: 20px;
          max-width: 400px;
          max-height: 60vh;
          overflow-y: auto;
          background: rgba(0,0,0,0.85);
          padding: 16px;
          border-radius: 8px;
          z-index: 1001;
        }
        .prompt-text {
          font-size: 0.85rem;
          color: rgba(255,255,255,0.9);
          line-height: 1.5;
          white-space: pre-wrap;
          word-break: break-word;
        }
        .prompt-label {
          font-size: 0.75rem;
          color: var(--accent-color, #a855f7);
          font-weight: 600;
          margin-bottom: 4px;
        }

        /* ========== BUTTONS & CONTROLS ========== */
        .delete-btn {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 12px;
          border-radius: 6px;
          border: none;
          background: #dc2626;
          color: #fff;
          font-size: 0.85rem;
          cursor: pointer;
        }
        .delete-btn:hover {
          background: #b91c1c;
        }
        .delete-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .header-btn {
          padding: 6px 10px;
          border-radius: 6px;
          border: none;
          background: rgba(255,255,255,0.1);
          color: var(--text-muted);
          font-size: 0.8rem;
          cursor: pointer;
        }
        .header-btn:hover {
          background: rgba(255,255,255,0.2);
        }
        .sort-select {
          padding: 6px 10px;
          border-radius: 6px;
          border: 1px solid var(--border-color);
          background: #1a1a1a;
          color: #e5e5e5;
          font-size: 0.8rem;
          cursor: pointer;
          outline: none;
        }
        .sort-select option {
          background: #1a1a1a;
          color: #e5e5e5;
        }
        .sort-btn {
          padding: 6px 8px;
          border-radius: 6px;
          border: none;
          background: rgba(255,255,255,0.1);
          color: var(--text-muted);
          cursor: pointer;
          display: flex;
          align-items: center;
        }
        .sort-btn:hover {
          background: rgba(255,255,255,0.2);
        }

        /* ========== ANIMATION ========== */
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}),r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"12px 16px",borderBottom:"1px solid var(--border-color)",backgroundColor:"var(--bg-secondary)",flexWrap:"wrap",gap:"10px"},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"16px"},children:[r.jsx("span",{style:{fontWeight:600,color:"var(--text-primary)"},children:e==="all"?"All Media":e==="video"?"Videos":e==="image"?"Images":e==="audio"?"Audio":"Prompts"}),r.jsxs("span",{style:{color:"var(--text-muted)",fontSize:"0.85rem"},children:[e==="prompts"?r.jsxs(r.Fragment,{children:["💬 ",pe.length," items with prompts"]}):r.jsxs(r.Fragment,{children:["🎬 ",p.videos," • 🖼️ ",p.images," • 🎵 ",p.audio," • ❤️ ",wn]}),L!=="all"&&` • 📋 ${pe.length} shown`]})]}),r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"4px",position:"relative"},children:[r.jsx(hx,{size:14,style:{color:"var(--text-muted)",position:"absolute",left:"8px"}}),r.jsx("input",{type:"text",placeholder:"Search filename or prompt...",value:T,onChange:S=>ne(S.target.value),style:{background:"rgba(255,255,255,0.08)",border:"1px solid var(--border-color)",borderRadius:"6px",padding:"6px 8px 6px 28px",color:"var(--text-primary)",fontSize:"0.85rem",width:"200px",outline:"none"}}),T&&r.jsx("button",{onClick:()=>ne(""),style:{position:"absolute",right:"6px",background:"none",border:"none",color:"var(--text-muted)",cursor:"pointer",padding:"2px"},children:r.jsx(Qe,{size:12})})]}),r.jsx("div",{style:{width:"1px",height:"20px",background:"var(--border-color)",margin:"0 4px"}}),r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"4px"},children:[r.jsx(Uh,{size:14,style:{color:"var(--text-muted)"}}),r.jsxs("select",{className:"sort-select",value:L,onChange:S=>{ee(S.target.value),w(new Set)},children:[r.jsx("option",{value:"all",children:"All"}),r.jsx("option",{value:"favorites",children:"❤️ Favorites"}),r.jsx("option",{value:"non-favorites",children:"🤍 Non-favorites"})]}),(e==="all"||e==="image")&&r.jsxs("button",{className:"sort-btn",onClick:()=>D(S=>!S),title:ae?"Click to show video source images":"Hiding video source images",style:{background:ae?void 0:"var(--accent-color, #a855f7)",color:ae?void 0:"#fff",fontSize:"0.75rem",padding:"4px 8px"},children:["📸",ae?"":"✓"]})]}),r.jsx("div",{style:{width:"1px",height:"20px",background:"var(--border-color)",margin:"0 4px"}}),r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"4px"},children:[r.jsx(lh,{size:14,style:{color:"var(--text-muted)"}}),r.jsxs("select",{className:"sort-select",value:G,onChange:S=>W(S.target.value),children:[r.jsx("option",{value:"date",children:"Date"}),r.jsx("option",{value:"name",children:"Name"}),r.jsx("option",{value:"size",children:"Size"}),r.jsx("option",{value:"favorites",children:"Favorites ❤️"}),r.jsx("option",{value:"non-favorites",children:"Non-favorites 🤍"})]}),r.jsx("button",{className:"sort-btn",onClick:()=>N(S=>S==="asc"?"desc":"asc"),title:b==="asc"?"Ascending":"Descending",children:b==="asc"?"↑":"↓"})]}),r.jsx("div",{style:{width:"1px",height:"20px",background:"var(--border-color)",margin:"0 4px"}}),r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"2px"},children:[r.jsx("span",{style:{color:"var(--text-muted)",fontSize:"0.75rem",marginRight:"4px"},children:"Profile:"}),["auto","1280x1024","1080p","1440p","4k"].map(S=>{var $,Z;return r.jsx("button",{className:"sort-btn",onClick:()=>{q(S),kl(S)},title:S==="auto"?`Auto-detect (currently ${Gc()})`:(($=Va[S])==null?void 0:$.label)||S,style:{background:U===S?"var(--accent-color, #a855f7)":void 0,color:U===S?"#fff":void 0,fontSize:"0.7rem",padding:"4px 6px"},children:S==="auto"?"⚡Auto":((Z=Va[S])==null?void 0:Z.label)||S},S)}),r.jsxs("span",{style:{color:"var(--text-muted)",fontSize:"0.7rem",marginLeft:"8px"},children:[Q," cols"]})]}),r.jsx("div",{style:{width:"1px",height:"20px",background:"var(--border-color)",margin:"0 4px"}}),k.size>0&&r.jsxs(r.Fragment,{children:[r.jsxs("span",{style:{color:"var(--text-muted)",fontSize:"0.85rem"},children:[k.size," selected"]}),r.jsx("button",{className:"header-btn",onClick:ya,children:"Clear"}),r.jsx("button",{className:"header-btn",onClick:bt,children:"Select All"}),r.jsxs("button",{className:"header-btn",onClick:bn,title:"Download selected items",children:[r.jsx(vt,{size:16}),"Download"]}),r.jsxs("button",{className:"delete-btn",onClick:yn,disabled:f,children:[r.jsx(Cs,{size:16}),f?"Deleting...":"Delete"]})]}),r.jsx("button",{onClick:Te,disabled:l,style:{padding:"8px",borderRadius:"6px",border:"none",background:"transparent",color:"var(--text-muted)",cursor:"pointer",display:"flex",alignItems:"center"},title:"Refresh",children:r.jsx(mn,{size:18,style:{animation:l?"spin 1s linear infinite":"none"}})}),r.jsx("button",{onClick:()=>Y(!0),style:{padding:"6px",border:"none",background:"transparent",color:"var(--text-muted)",cursor:"pointer",display:"flex",alignItems:"center"},title:"Keyboard shortcuts (?)",children:r.jsx(ip,{size:18})})]})]}),C&&r.jsx("div",{style:{position:"fixed",top:0,left:0,right:0,bottom:0,backgroundColor:"rgba(0,0,0,0.8)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:2e3},onClick:()=>Y(!1),children:r.jsxs("div",{style:{backgroundColor:"var(--bg-primary, #1a1a1a)",borderRadius:"12px",padding:"24px",maxWidth:"500px",width:"90%",boxShadow:"0 20px 60px rgba(0,0,0,0.5)"},onClick:S=>S.stopPropagation(),children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"20px"},children:[r.jsx("h3",{style:{margin:0,color:"var(--text-primary, #fff)",fontSize:"1.2rem"},children:"⌨️ Keyboard Shortcuts"}),r.jsx("button",{onClick:()=>Y(!1),style:{background:"transparent",border:"none",color:"var(--text-muted)",cursor:"pointer",padding:"4px"},children:r.jsx(Qe,{size:20})})]}),r.jsxs("div",{style:{color:"var(--text-secondary, #ccc)",fontSize:"0.9rem"},children:[r.jsxs("div",{style:{marginBottom:"16px"},children:[r.jsx("div",{style:{color:"var(--accent-color, #a855f7)",fontWeight:600,marginBottom:"8px"},children:"Grid View"}),r.jsxs("div",{style:{display:"grid",gridTemplateColumns:"120px 1fr",gap:"6px 16px"},children:[r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"+"}),r.jsx("span",{children:"More columns (smaller thumbnails)"}),r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"-"}),r.jsx("span",{children:"Fewer columns (larger thumbnails)"}),r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"?"}),r.jsx("span",{children:"Show this help"})]})]}),r.jsxs("div",{style:{marginBottom:"16px"},children:[r.jsx("div",{style:{color:"var(--accent-color, #a855f7)",fontWeight:600,marginBottom:"8px"},children:"Lightbox (Image View)"}),r.jsxs("div",{style:{display:"grid",gridTemplateColumns:"120px 1fr",gap:"6px 16px"},children:[r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"←"}),r.jsx("span",{children:"Previous image"}),r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"→"}),r.jsx("span",{children:"Next image"}),r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"F / H"}),r.jsx("span",{children:"Toggle favorite ❤️"}),r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"Esc"}),r.jsx("span",{children:"Close lightbox"})]})]}),r.jsxs("div",{children:[r.jsx("div",{style:{color:"var(--accent-color, #a855f7)",fontWeight:600,marginBottom:"8px"},children:"Selection"}),r.jsxs("div",{style:{display:"grid",gridTemplateColumns:"120px 1fr",gap:"6px 16px"},children:[r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"Ctrl+Click"}),r.jsx("span",{children:"Toggle single item"}),r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"Shift+Click"}),r.jsx("span",{children:"Select range"})]})]})]}),r.jsx("div",{style:{marginTop:"20px",paddingTop:"16px",borderTop:"1px solid var(--border-color, #333)",textAlign:"center"},children:r.jsxs("span",{style:{color:"var(--text-muted)",fontSize:"0.8rem"},children:["Press ",r.jsx("kbd",{style:{background:"#333",padding:"2px 6px",borderRadius:"4px"},children:"?"})," or ",r.jsx("kbd",{style:{background:"#333",padding:"2px 6px",borderRadius:"4px"},children:"Esc"})," to close"]})})]})}),c&&r.jsx("div",{style:{padding:"12px 16px",backgroundColor:"rgba(239, 68, 68, 0.1)",color:"#ef4444",textAlign:"center"},children:c}),l&&r.jsxs("div",{style:{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",color:"var(--text-muted)"},children:[r.jsx(mn,{size:40,style:{animation:"spin 1s linear infinite",marginBottom:"16px"}}),r.jsx("div",{children:"Loading media..."})]}),!l&&a.length===0&&r.jsxs("div",{style:{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",color:"var(--text-muted)"},children:[r.jsx("div",{style:{fontSize:"4rem",marginBottom:"16px",opacity:.5},children:"📁"}),r.jsxs("div",{style:{fontSize:"1.2rem",marginBottom:"8px"},children:["No ",e==="prompts"?"prompts":e==="all"?"media":e+"s"," yet"]}),r.jsx("div",{style:{fontSize:"0.9rem",opacity:.7},children:"Generated content will appear here"})]}),!l&&pe.length>0&&e==="prompts"&&r.jsx("div",{ref:te,className:"prompts-list",onScroll:K,style:{flex:1,overflowY:"auto",overflowX:"hidden",padding:"16px",display:"flex",flexDirection:"column",gap:"12px"},children:pe.slice(0,M).map((S,$)=>{var Z,ie,ye,ce;return r.jsxs("div",{style:{display:"flex",gap:"16px",padding:"16px",backgroundColor:"var(--bg-secondary, #1f1f1f)",borderRadius:"12px",border:"1px solid var(--border-color, #333)",cursor:"pointer",transition:"border-color 0.15s"},onClick:()=>x($),onMouseEnter:me=>me.currentTarget.style.borderColor="var(--accent-color, #a855f7)",onMouseLeave:me=>me.currentTarget.style.borderColor="var(--border-color, #333)",children:[r.jsx("div",{style:{flexShrink:0},children:S.type==="video"?r.jsx("video",{src:`${oe}${S.url}`,style:{width:"100px",height:"100px",objectFit:"cover",borderRadius:"8px"},autoPlay:!0,loop:!0,muted:!0,playsInline:!0}):r.jsx("img",{src:`${oe}${S.url}`,alt:S.filename,style:{width:"100px",height:"100px",objectFit:"cover",borderRadius:"8px"},loading:"lazy"})}),r.jsxs("div",{style:{flex:1,minWidth:0},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:"8px"},children:[r.jsxs("div",{children:[r.jsx("div",{style:{fontSize:"0.85rem",fontWeight:600,color:"var(--text-primary)",marginBottom:"4px"},children:S.filename}),r.jsxs("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:[S.type==="video"?"🎬":S.type==="audio"?"🎵":"🖼️"," ",It(S.size),((Z=S.metadata)==null?void 0:Z.steps)&&` • ${S.metadata.steps} steps`,((ie=S.metadata)==null?void 0:ie.cfg)&&` • CFG ${S.metadata.cfg}`]})]}),r.jsxs("div",{style:{display:"flex",gap:"8px"},children:[r.jsxs("button",{style:{background:"var(--accent-color, #a855f7)",border:"none",color:"#fff",padding:"6px 12px",borderRadius:"6px",cursor:"pointer",fontSize:"0.75rem",display:"flex",alignItems:"center",gap:"4px"},onClick:me=>{var Zt,E;me.stopPropagation();const at=((Zt=S.metadata)==null?void 0:Zt.positive_prompt)||((E=S.metadata)==null?void 0:E.prompt);navigator.clipboard.writeText(at)},children:[r.jsx(Wt,{size:12}),"Copy"]}),r.jsx("button",{className:(_.has(S.filename),""),style:{background:_.has(S.filename)?"#ef4444":"rgba(255,255,255,0.1)",border:"none",color:"#fff",padding:"6px",borderRadius:"6px",cursor:"pointer"},onClick:me=>de(S.filename,me),children:r.jsx(yl,{size:14,fill:_.has(S.filename)?"#fff":"none"})})]})]}),r.jsx("div",{style:{fontSize:"0.9rem",color:"var(--text-primary)",lineHeight:1.5,backgroundColor:"var(--bg-tertiary, #2a2a2a)",padding:"10px 12px",borderRadius:"6px",maxHeight:"100px",overflow:"hidden",textOverflow:"ellipsis",display:"-webkit-box",WebkitLineClamp:4,WebkitBoxOrient:"vertical"},children:((ye=S.metadata)==null?void 0:ye.positive_prompt)||((ce=S.metadata)==null?void 0:ce.prompt)})]})]},S.filename)})}),!l&&pe.length>0&&e!=="prompts"&&r.jsx("div",{ref:te,className:"media-grid",onScroll:K,style:{flex:1,overflowY:"auto",overflowX:"hidden",gridTemplateColumns:`repeat(${Q}, 1fr)`},children:pe.slice(0,M).map((S,$)=>{var Z,ie,ye;return r.jsxs("div",{className:`thumb-card ${k.has($)?"selected":""}`,style:{height:`${A}px`},onClick:ce=>nt($,ce),children:[r.jsx("div",{className:"select-checkbox",onClick:ce=>Pt($,ce),children:k.has($)&&r.jsx(Ns,{size:14,color:"#fff"})}),r.jsx("div",{className:`favorite-btn ${_.has(S.filename)?"is-favorite":""}`,onClick:ce=>de(S.filename,ce),title:_.has(S.filename)?"Remove from favorites":"Add to favorites",children:r.jsx(yl,{size:14,color:_.has(S.filename)?"#fff":"rgba(255,255,255,0.7)",fill:_.has(S.filename)?"#fff":"none"})}),(((Z=S.metadata)==null?void 0:Z.positive_prompt)||((ie=S.metadata)==null?void 0:ie.prompt))&&r.jsx("button",{className:"prompt-bubble-btn",onClick:ce=>{ce.stopPropagation(),I({item:S})},title:"View prompt",children:"💬"}),S.has_source_image&&r.jsxs("div",{className:"source-image-badge",children:[r.jsx(gr,{size:10}),r.jsx("span",{children:"+IMG"})]}),S.type==="video"?r.jsx("video",{src:`${oe}${S.url}`,autoPlay:!0,loop:!0,muted:!0,playsInline:!0,preload:"metadata",onLoadedMetadata:ce=>{const me=ce.target.duration;me&&!P[S.filename]&&O(at=>({...at,[S.filename]:me}))}}):S.type==="audio"?r.jsxs("div",{className:"audio-thumb",children:[r.jsx("div",{className:"audio-icon",children:"🎵"}),r.jsx("audio",{src:`${oe}${S.url}`,preload:"metadata",onLoadedMetadata:ce=>{const me=ce.target.duration;me&&!P[S.filename]&&O(at=>({...at,[S.filename]:me}))}})]}):r.jsx("img",{src:`${oe}${S.url}`,alt:S.filename,loading:"lazy"}),r.jsxs("div",{className:"media-overlay",children:[r.jsxs("div",{children:[r.jsx("div",{className:"media-filename",children:S.filename}),r.jsxs("div",{className:"media-size",children:[It(S.size),(S.type==="video"||S.type==="audio")&&P[S.filename]&&r.jsxs("span",{className:"media-duration",children:[r.jsx(Bs,{size:10}),Hc(P[S.filename])]})]})]}),r.jsxs("div",{className:"overlay-buttons",children:[((ye=S.metadata)==null?void 0:ye.has_metadata)&&r.jsx("button",{className:"overlay-btn",onClick:ce=>ja(S,ce),title:"Download metadata JSON",children:r.jsx(Ic,{size:14})}),r.jsx("button",{className:"overlay-btn",onClick:ce=>jn(S,ce),title:"Download",children:r.jsx(vt,{size:14})})]})]})]},S.filename)})}),ve&&r.jsxs("div",{className:"lightbox-overlay",onClick:()=>x(null),children:[r.jsx("button",{className:"lightbox-close",onClick:()=>x(null),children:r.jsx(Qe,{size:24})}),((ba=ve.metadata)==null?void 0:ba.has_metadata)&&r.jsx("button",{style:{position:"absolute",top:"20px",left:"20px",padding:"8px 12px",borderRadius:"6px",background:h?"var(--accent-color, #a855f7)":"rgba(255,255,255,0.1)",border:"none",color:"#fff",cursor:"pointer",fontSize:"0.85rem",zIndex:1002},onClick:S=>{S.stopPropagation(),y(!h)},children:h?"Hide Prompt":"Show Prompt"}),h&&ve.metadata&&r.jsxs("div",{className:"lightbox-metadata",onClick:S=>S.stopPropagation(),children:[ve.metadata.positive_prompt&&r.jsxs("div",{style:{marginBottom:"16px"},children:[r.jsx("div",{className:"prompt-label",children:"✨ Positive Prompt"}),r.jsx("div",{className:"prompt-text",children:ve.metadata.positive_prompt})]}),ve.metadata.negative_prompt&&r.jsxs("div",{children:[r.jsx("div",{className:"prompt-label",children:"🚫 Negative Prompt"}),r.jsx("div",{className:"prompt-text",style:{color:"rgba(255,255,255,0.6)"},children:ve.metadata.negative_prompt})]})]}),r.jsx("button",{className:"lightbox-nav",style:{left:"20px"},onClick:S=>{S.stopPropagation(),x($=>$>0?$-1:pe.length-1)},children:r.jsx(lp,{size:28})}),r.jsx("div",{className:"lightbox-content",onClick:S=>S.stopPropagation(),children:ve.type==="video"?r.jsx("video",{src:`${oe}${ve.url}`,autoPlay:!0,loop:!0,controls:!0,style:{borderRadius:"12px"}}):ve.type==="audio"?r.jsxs("div",{className:"audio-lightbox",children:[r.jsx("div",{className:"audio-icon-large",children:"🎵"}),r.jsx("div",{className:"audio-filename",children:ve.filename}),r.jsx("audio",{src:`${oe}${ve.url}`,autoPlay:!0,controls:!0,style:{width:"100%",maxWidth:"400px",marginTop:"20px"}})]}):r.jsx("img",{src:`${oe}${ve.url}`,alt:ve.filename,style:{borderRadius:"12px"}})}),r.jsx("button",{className:"lightbox-nav",style:{right:"20px"},onClick:S=>{S.stopPropagation(),x($=>$<pe.length-1?$+1:0)},children:r.jsx(op,{size:28})}),r.jsxs("div",{className:"lightbox-info",children:[r.jsx("span",{style:{color:"#fff",fontWeight:500},children:ve.filename}),r.jsx("span",{style:{color:"rgba(255,255,255,0.6)"},children:It(ve.size)}),_.has(ve.filename)&&r.jsx("span",{style:{color:"#ef4444",fontSize:"0.8rem"},children:"❤️ Favorite"}),ve.has_source_image&&r.jsx("span",{style:{color:"#3b82f6",fontSize:"0.8rem"},children:"📷 Has source image"}),r.jsxs("span",{style:{color:"rgba(255,255,255,0.5)"},children:[g+1," / ",pe.length]}),r.jsxs("div",{style:{display:"flex",gap:"8px"},children:[r.jsx("button",{className:"overlay-btn",onClick:S=>de(ve.filename,S),title:_.has(ve.filename)?"Remove from favorites":"Add to favorites",style:{background:_.has(ve.filename)?"rgba(239, 68, 68, 0.5)":void 0},children:r.jsx(yl,{size:16,fill:_.has(ve.filename)?"#ef4444":"none",color:_.has(ve.filename)?"#ef4444":"#fff"})}),ve.has_source_image&&ve.source_image&&r.jsx("button",{className:"overlay-btn",onClick:S=>jn(ve.source_image,S),title:"Download source image",children:r.jsx(gr,{size:16})}),((kn=ve.metadata)==null?void 0:kn.has_metadata)&&r.jsx("button",{className:"overlay-btn",onClick:S=>ja(ve,S),title:"Download metadata JSON",children:r.jsx(Ic,{size:16})}),r.jsx("button",{className:"overlay-btn",onClick:S=>jn(ve,S),title:"Download",children:r.jsx(vt,{size:16})})]})]})]}),j&&r.jsx("div",{className:"prompt-popup-overlay",onClick:()=>I(null),children:r.jsxs("div",{className:"prompt-popup",onClick:S=>S.stopPropagation(),children:[r.jsxs("div",{className:"prompt-popup-header",children:[r.jsxs("div",{className:"prompt-popup-title",children:[r.jsx(rx,{size:18}),"Prompt Details"]}),r.jsx("button",{className:"prompt-popup-close",onClick:()=>I(null),children:r.jsx(Qe,{size:20})})]}),r.jsxs("div",{className:"prompt-popup-content",children:[r.jsxs("div",{style:{display:"flex",gap:"12px",alignItems:"flex-start"},children:[j.item.type==="video"?r.jsx("video",{src:`${oe}${j.item.url}`,className:"prompt-media-preview",autoPlay:!0,loop:!0,muted:!0,playsInline:!0}):r.jsx("img",{src:`${oe}${j.item.url}`,alt:j.item.filename,className:"prompt-media-preview"}),r.jsxs("div",{style:{flex:1},children:[r.jsx("div",{style:{fontSize:"0.85rem",fontWeight:600,color:"var(--text-primary)"},children:j.item.filename}),r.jsxs("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"4px"},children:[j.item.type==="video"?"🎬 Video":"🖼️ Image"," • ",It(j.item.size),j.item.type==="video"&&P[j.item.filename]&&r.jsxs(r.Fragment,{children:[" • ",Hc(P[j.item.filename])]}),((wa=j.item.metadata)==null?void 0:wa.width)&&((Hs=j.item.metadata)==null?void 0:Hs.height)&&r.jsxs(r.Fragment,{children:[" • ",j.item.metadata.width,"×",j.item.metadata.height]})]})]})]}),(((ka=j.item.metadata)==null?void 0:ka.positive_prompt)||((kr=j.item.metadata)==null?void 0:kr.prompt))&&r.jsxs("div",{className:"prompt-section",children:[r.jsx("div",{className:"prompt-section-label",children:"✨ Positive Prompt"}),r.jsx("div",{className:"prompt-section-text",children:j.item.metadata.positive_prompt||j.item.metadata.prompt}),r.jsxs("button",{className:"prompt-copy-btn",onClick:()=>{const S=j.item.metadata.positive_prompt||j.item.metadata.prompt;navigator.clipboard.writeText(S)},children:[r.jsx(Wt,{size:14}),"Copy Prompt"]})]}),((Sn=j.item.metadata)==null?void 0:Sn.negative_prompt)&&r.jsxs("div",{className:"prompt-section",children:[r.jsx("div",{className:"prompt-section-label",children:"🚫 Negative Prompt"}),r.jsx("div",{className:"prompt-section-text",style:{color:"var(--text-muted)"},children:j.item.metadata.negative_prompt})]}),(((Nn=j.item.metadata)==null?void 0:Nn.steps)||((wt=j.item.metadata)==null?void 0:wt.cfg)||((Gs=j.item.metadata)==null?void 0:Gs.seed)||((Ar=j.item.metadata)==null?void 0:Ar.sampler)||((Sa=j.item.metadata)==null?void 0:Sa.model))&&r.jsxs("div",{className:"prompt-section",children:[r.jsx("div",{className:"prompt-section-label",children:"⚙️ Generation Settings"}),r.jsxs("div",{style:{display:"flex",gap:"12px",flexWrap:"wrap",fontSize:"0.85rem"},children:[j.item.metadata.steps&&r.jsxs("span",{children:["Steps: ",r.jsx("strong",{children:j.item.metadata.steps})]}),j.item.metadata.cfg&&r.jsxs("span",{children:["CFG: ",r.jsx("strong",{children:j.item.metadata.cfg})]}),j.item.metadata.seed&&r.jsxs("span",{children:["Seed: ",r.jsx("strong",{children:j.item.metadata.seed})]}),j.item.metadata.sampler&&r.jsxs("span",{children:["Sampler: ",r.jsx("strong",{children:j.item.metadata.sampler})]}),j.item.metadata.scheduler&&r.jsxs("span",{children:["Scheduler: ",r.jsx("strong",{children:j.item.metadata.scheduler})]})]}),j.item.metadata.model&&r.jsxs("div",{style:{marginTop:"8px",fontSize:"0.8rem",color:"var(--text-muted)"},children:["Model: ",r.jsx("strong",{style:{color:"var(--text-primary)"},children:j.item.metadata.model})]})]}),((ft=j.item.metadata)==null?void 0:ft.loras)&&j.item.metadata.loras.length>0&&r.jsxs("div",{className:"prompt-section",children:[r.jsx("div",{className:"prompt-section-label",children:"🎨 LoRAs Used"}),r.jsx("div",{style:{display:"flex",flexDirection:"column",gap:"6px",fontSize:"0.85rem"},children:j.item.metadata.loras.map((S,$)=>r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"6px 10px",backgroundColor:"var(--bg-secondary)",borderRadius:"4px"},children:[r.jsx("span",{style:{fontFamily:"monospace",fontSize:"0.8rem",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:"80%"},children:S.name}),r.jsxs("span",{style:{color:"var(--accent-color, #a855f7)",fontWeight:600,fontSize:"0.8rem"},children:[(S.strength*100).toFixed(0),"%"]})]},$))})]})]})]})})]})}const Wg=()=>{const e=oe.startsWith("https")?"wss:":"ws:",t=oe.replace(/^https?:\/\//,"");return`${e}//${t}/ws/logs`};function Hg(){const[e,t]=i.useState([]),[n,a]=i.useState(!0),[s,l]=i.useState(!1),[o,c]=i.useState(!1),d=i.useRef(null),p=i.useRef(null),v=i.useRef(null),g=i.useCallback(()=>{var k;if(((k=p.current)==null?void 0:k.readyState)===WebSocket.OPEN)return;const x=new WebSocket(Wg());p.current=x,x.onopen=()=>{c(!0),console.log("📡 Log WebSocket connected")},x.onmessage=w=>{try{const z=JSON.parse(w.data);t(F=>[...F,z].slice(-500))}catch(z){console.error("Failed to parse log",z)}},x.onclose=()=>{c(!1),console.log("📡 Log WebSocket disconnected"),v.current=setTimeout(()=>{n&&g()},3e3)},x.onerror=w=>{console.error("WebSocket error",w),x.close()}},[n]);return i.useEffect(()=>{var x;return n?g():((x=p.current)==null||x.close(),v.current&&clearTimeout(v.current)),()=>{var k;(k=p.current)==null||k.close(),v.current&&clearTimeout(v.current)}},[n,g]),i.useEffect(()=>{d.current&&d.current.scrollIntoView({behavior:"smooth"})},[e]),n?r.jsxs("div",{style:{position:"fixed",bottom:"20px",right:"20px",width:s?"800px":"400px",height:s?"600px":"300px",backgroundColor:"#0a0a0a",border:"1px solid #333",borderRadius:"8px",display:"flex",flexDirection:"column",zIndex:100,boxShadow:"0 10px 30px rgba(0,0,0,0.8)",transition:"all 0.2s ease"},children:[r.jsxs("div",{style:{padding:"8px 12px",borderBottom:"1px solid #333",display:"flex",justifyContent:"space-between",alignItems:"center",backgroundColor:"#121212",borderTopLeftRadius:"8px",borderTopRightRadius:"8px"},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px",fontSize:"0.8rem",fontWeight:600,color:"#a3a3a3"},children:[r.jsx(Mc,{size:14}),r.jsx("span",{children:"Server Logs"}),o?r.jsx(Mx,{size:12,color:"#22c55e",title:"Connected"}):r.jsx(Ix,{size:12,color:"#ef4444",title:"Disconnected"})]}),r.jsxs("div",{style:{display:"flex",gap:"8px"},children:[r.jsx("button",{onClick:()=>l(!s),style:{background:"transparent",border:"none",cursor:"pointer",color:"#666"},children:s?r.jsx(lx,{size:14}):r.jsx(up,{size:14})}),r.jsx("button",{onClick:()=>a(!1),style:{background:"transparent",border:"none",cursor:"pointer",color:"#666"},children:r.jsx(Qe,{size:14})})]})]}),r.jsxs("div",{style:{flex:1,overflowY:"auto",padding:"12px",fontFamily:"monospace",fontSize:"0.75rem",color:"#d4d4d4",lineHeight:"1.4"},children:[e.map((x,k)=>{var w,z;return r.jsxs("div",{style:{marginBottom:"4px",display:"flex",gap:"8px"},children:[r.jsx("span",{style:{color:"#525252",flexShrink:0},children:((z=(w=x.timestamp)==null?void 0:w.split("T")[1])==null?void 0:z.split(".")[0])||""}),r.jsx("span",{style:{color:x.level==="ERROR"?"#ef4444":x.level==="WARNING"?"#eab308":"#a3a3a3"},children:x.message})]},k)}),r.jsx("div",{ref:d})]})]}):r.jsx("button",{onClick:()=>a(!0),style:{position:"fixed",bottom:"20px",right:"20px",backgroundColor:"#1a1a1a",border:"1px solid #333",borderRadius:"50%",width:"48px",height:"48px",display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",zIndex:100,boxShadow:"0 4px 12px rgba(0,0,0,0.5)"},children:r.jsx(Mc,{size:20,color:"#a3a3a3"})})}function Gg(){const[e,t]=i.useState(J.IMAGE_TO_VIDEO),[n,a]=i.useState(!1),[s,l]=i.useState(null),[o,c]=i.useState(!1),[d,p]=i.useState(null),[v,g]=i.useState(0),[x,k]=i.useState(0),[w,z]=i.useState(!1),[F,f]=i.useState(null),u=i.useRef(null),h=async()=>{try{const G=await(await fetch(`${oe}/health`)).json();l(G)}catch{l(null)}};i.useEffect(()=>{h();const R=setInterval(h,1e4);return()=>clearInterval(R)},[]);const y=async()=>{if(!o&&window.confirm("Backend herstarten? Lopende jobs worden afgebroken.")){c(!0);try{await fetch(`${oe}/restart`,{method:"POST"}),await new Promise(R=>setTimeout(R,3e3)),await h()}catch(R){console.error("Restart failed:",R)}finally{c(!1)}}},j=()=>{const R=u.current;if(!R){alert("Geen parameters beschikbaar");return}const G=new Blob([JSON.stringify(R,null,2)],{type:"application/json"}),W=URL.createObjectURL(G),b=document.createElement("a");b.href=W,b.download=`${e}_params_${Date.now()}.json`,b.click(),URL.revokeObjectURL(W)},I=i.useMemo(()=>{switch(e){case J.TEXT_TO_VIDEO:return"Text to Video";case J.IMAGE_TO_VIDEO:return"Image to Video";case J.TEXT_TO_IMAGE_TO_VIDEO:return"Text to Image to Video";case J.VIDEO_TO_VIDEO:return"Video to Video";case J.VIDEO_TO_TEXT:return"Video to Text";case J.PIPELINE:return"Pipeline";case J.LORA_TRAINING:return"LoRA Training";case J.TEXT_TO_IMAGE:return"Text to Image";case J.IMAGE_TO_IMAGE:return"Image to Image";case J.REFRAME:return"Reframe";case J.FACE_SWAP:return"Face Swap";case J.UPSCALER:return"Upscaler";case J.IMAGE_TO_TEXT:return"Image to Text";case J.PROMPT_GENERATOR:return"Prompt Generator";case J.AUDIO_GENERATION:return"Audio Generation";case J.VOICE_CLONING:return"Voice Cloning";case J.LIP_SYNC:return"Lip Sync";case J.SPEECH_TO_VIDEO:return"Speech to Video";case J.MY_MEDIA_ALL:return"My Media - All";case J.MY_MEDIA_VIDEOS:return"My Media - Videos";case J.MY_MEDIA_IMAGES:return"My Media - Images";case J.MY_MEDIA_PROMPTS:return"My Media - Prompts";default:return"Tool"}},[e]),_=()=>{const R=()=>g(N=>N+1),G=(N,L)=>{z(N),f(()=>L)},W=N=>{u.current=N},b=()=>{k(N=>N+1)};switch(e){case J.TEXT_TO_VIDEO:return r.jsx(Jx,{onOutput:p,onRefreshHistory:R,onParamsChange:W,onJobSubmitted:b});case J.IMAGE_TO_VIDEO:return r.jsx(sg,{onOutput:p,onRefreshHistory:R,onCreationsModeChange:G,onParamsChange:W,onJobSubmitted:b});case J.TEXT_TO_IMAGE_TO_VIDEO:return r.jsx(og,{onOutput:p,onParamsChange:W,onJobSubmitted:b});case J.PIPELINE:return r.jsx(yg,{});case J.LORA_TRAINING:return r.jsx(jg,{onOutput:p});case J.MY_MEDIA_ALL:return r.jsx(Ur,{filter:"all"});case J.MY_MEDIA_VIDEOS:return r.jsx(Ur,{filter:"video"});case J.MY_MEDIA_IMAGES:return r.jsx(Ur,{filter:"image"});case J.MY_MEDIA_AUDIO:return r.jsx(Ur,{filter:"audio"});case J.MY_MEDIA_PROMPTS:return r.jsx(Ur,{filter:"prompts"});case J.TEXT_TO_IMAGE:return r.jsx(lg,{onOutput:p,onJobSubmitted:b});case J.IMAGE_TO_TEXT:return r.jsx(kg,{});case J.PROMPT_GENERATOR:return r.jsx(Sg,{});case J.IMAGE_TO_IMAGE:return r.jsx(Ng,{onOutput:p,onJobSubmitted:b});case J.UPSCALER:return r.jsx(_g,{onOutput:p,onJobSubmitted:b});case J.VIDEO_TO_VIDEO:return r.jsx(dg,{onOutput:p,onJobSubmitted:b});case J.VIDEO_TO_TEXT:return r.jsx(mg,{});case J.AUDIO_GENERATION:return r.jsx(Tg,{onOutput:p,onJobSubmitted:b});case J.VOICE_CLONING:return r.jsx(Ig,{onOutput:p,onJobSubmitted:b});case J.LIP_SYNC:return r.jsx(Fg,{onOutput:p,onJobSubmitted:b});case J.SPEECH_TO_VIDEO:return r.jsx(vg,{onOutput:p,onJobSubmitted:b});case J.REFRAME:return r.jsx(Dg,{onOutput:p,onJobSubmitted:b});case J.FACE_SWAP:return r.jsx(Ag,{onOutput:p,onJobSubmitted:b});default:return r.jsx($g,{title:I})}};return r.jsxs("div",{className:"dashboard-container",children:[r.jsx(qx,{health:s,onRestartBackend:y,restarting:o}),r.jsxs("div",{className:"dashboard-body",children:[r.jsx(Bx,{activeToolId:e,onSelectTool:t,collapsed:n,onToggleCollapsed:()=>a(R=>!R)}),r.jsxs("main",{className:"main-content",children:[r.jsxs("div",{className:"top-bar",children:[r.jsx("h1",{children:I}),r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"12px"},children:[r.jsx(Gx,{refreshToken:x,onJobComplete:R=>{g(G=>G+1),R.output_video&&p({kind:"video",url:`${oe}${R.output_video}`,backendUrl:`${oe}${R.output_video}`})}}),r.jsx("button",{className:"icon-btn",onClick:y,disabled:o,title:"Herstart Backend",style:{opacity:o?.5:1},children:r.jsx(mn,{size:18,color:"#fbbf24",className:o?"spin":""})}),r.jsxs("div",{className:"status-indicator",children:[r.jsx("div",{className:`status-dot ${(s==null?void 0:s.status)==="healthy"?"connected":""}`}),r.jsx("span",{children:(s==null?void 0:s.status)==="healthy"?"Connected":"Disconnected"})]})]})]}),e===J.MY_MEDIA_ALL||e===J.MY_MEDIA_VIDEOS||e===J.MY_MEDIA_IMAGES||e===J.MY_MEDIA_AUDIO||e===J.MY_MEDIA_PROMPTS?r.jsx("div",{style:{flex:1,display:"flex",flexDirection:"column",overflow:"hidden"},children:_()}):r.jsxs("div",{className:"workspace",children:[r.jsxs("section",{className:"controls-panel",children:[r.jsxs("div",{className:"panel-header",style:{marginBottom:"16px",display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsx("div",{className:"panel-title",style:{fontSize:"0.85rem",fontWeight:600,color:"var(--text-secondary)",textTransform:"uppercase",letterSpacing:"0.05em"},children:"Parameters"}),r.jsx("button",{className:"icon-btn",onClick:j,title:"Download parameters als JSON",style:{padding:"4px"},children:r.jsx(vt,{size:16})})]}),r.jsx("div",{className:"panel-body",children:_()})]}),d?r.jsx(Hx,{output:d,refreshToken:v,onSelectHistoryVideo:p,onClose:()=>p(null)}):r.jsxs("section",{className:"output-panel",style:{display:"flex",flexDirection:"column"},children:[w&&r.jsxs("div",{style:{padding:"12px 16px",borderBottom:"1px solid var(--border-color)",backgroundColor:"var(--bg-secondary)",display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsx("span",{style:{fontWeight:600,color:"var(--text-primary)"},children:"Select Image for I2V"}),r.jsx("span",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:"Click an image to use it"})]}),r.jsx("div",{style:{flex:1,overflow:"hidden"},children:r.jsx(Ur,{filter:"all",selectionMode:w,onSelectItem:F})})]})]})]})]}),r.jsx(Hg,{})]})}function Qg(){return r.jsx(Qx,{children:r.jsx(Gg,{})})}Sl.createRoot(document.getElementById("root")).render(r.jsx(Ap.StrictMode,{children:r.jsx(Qg,{})}));
