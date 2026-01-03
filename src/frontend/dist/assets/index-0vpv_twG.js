(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))s(a);new MutationObserver(a=>{for(const l of a)if(l.type==="childList")for(const o of l.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&s(o)}).observe(document,{childList:!0,subtree:!0});function n(a){const l={};return a.integrity&&(l.integrity=a.integrity),a.referrerPolicy&&(l.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?l.credentials="include":a.crossOrigin==="anonymous"?l.credentials="omit":l.credentials="same-origin",l}function s(a){if(a.ep)return;a.ep=!0;const l=n(a);fetch(a.href,l)}})();function np(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}var Tc={exports:{}},va={},Pc={exports:{}},ue={};/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var is=Symbol.for("react.element"),sp=Symbol.for("react.portal"),ap=Symbol.for("react.fragment"),lp=Symbol.for("react.strict_mode"),op=Symbol.for("react.profiler"),ip=Symbol.for("react.provider"),cp=Symbol.for("react.context"),dp=Symbol.for("react.forward_ref"),up=Symbol.for("react.suspense"),pp=Symbol.for("react.memo"),fp=Symbol.for("react.lazy"),ii=Symbol.iterator;function mp(e){return e===null||typeof e!="object"?null:(e=ii&&e[ii]||e["@@iterator"],typeof e=="function"?e:null)}var Ic={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},Mc=Object.assign,Rc={};function un(e,t,n){this.props=e,this.context=t,this.refs=Rc,this.updater=n||Ic}un.prototype.isReactComponent={};un.prototype.setState=function(e,t){if(typeof e!="object"&&typeof e!="function"&&e!=null)throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,e,t,"setState")};un.prototype.forceUpdate=function(e){this.updater.enqueueForceUpdate(this,e,"forceUpdate")};function Lc(){}Lc.prototype=un.prototype;function io(e,t,n){this.props=e,this.context=t,this.refs=Rc,this.updater=n||Ic}var co=io.prototype=new Lc;co.constructor=io;Mc(co,un.prototype);co.isPureReactComponent=!0;var ci=Array.isArray,Fc=Object.prototype.hasOwnProperty,uo={current:null},Dc={key:!0,ref:!0,__self:!0,__source:!0};function Oc(e,t,n){var s,a={},l=null,o=null;if(t!=null)for(s in t.ref!==void 0&&(o=t.ref),t.key!==void 0&&(l=""+t.key),t)Fc.call(t,s)&&!Dc.hasOwnProperty(s)&&(a[s]=t[s]);var i=arguments.length-2;if(i===1)a.children=n;else if(1<i){for(var d=Array(i),u=0;u<i;u++)d[u]=arguments[u+2];a.children=d}if(e&&e.defaultProps)for(s in i=e.defaultProps,i)a[s]===void 0&&(a[s]=i[s]);return{$$typeof:is,type:e,key:l,ref:o,props:a,_owner:uo.current}}function hp(e,t){return{$$typeof:is,type:e.type,key:t,ref:e.ref,props:e.props,_owner:e._owner}}function po(e){return typeof e=="object"&&e!==null&&e.$$typeof===is}function xp(e){var t={"=":"=0",":":"=2"};return"$"+e.replace(/[=:]/g,function(n){return t[n]})}var di=/\/+/g;function Oa(e,t){return typeof e=="object"&&e!==null&&e.key!=null?xp(""+e.key):t.toString(36)}function Ls(e,t,n,s,a){var l=typeof e;(l==="undefined"||l==="boolean")&&(e=null);var o=!1;if(e===null)o=!0;else switch(l){case"string":case"number":o=!0;break;case"object":switch(e.$$typeof){case is:case sp:o=!0}}if(o)return o=e,a=a(o),e=s===""?"."+Oa(o,0):s,ci(a)?(n="",e!=null&&(n=e.replace(di,"$&/")+"/"),Ls(a,t,n,"",function(u){return u})):a!=null&&(po(a)&&(a=hp(a,n+(!a.key||o&&o.key===a.key?"":(""+a.key).replace(di,"$&/")+"/")+e)),t.push(a)),1;if(o=0,s=s===""?".":s+":",ci(e))for(var i=0;i<e.length;i++){l=e[i];var d=s+Oa(l,i);o+=Ls(l,t,n,d,a)}else if(d=mp(e),typeof d=="function")for(e=d.call(e),i=0;!(l=e.next()).done;)l=l.value,d=s+Oa(l,i++),o+=Ls(l,t,n,d,a);else if(l==="object")throw t=String(e),Error("Objects are not valid as a React child (found: "+(t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t)+"). If you meant to render a collection of children, use an array instead.");return o}function gs(e,t,n){if(e==null)return e;var s=[],a=0;return Ls(e,s,"","",function(l){return t.call(n,l,a++)}),s}function gp(e){if(e._status===-1){var t=e._result;t=t(),t.then(function(n){(e._status===0||e._status===-1)&&(e._status=1,e._result=n)},function(n){(e._status===0||e._status===-1)&&(e._status=2,e._result=n)}),e._status===-1&&(e._status=0,e._result=t)}if(e._status===1)return e._result.default;throw e._result}var He={current:null},Fs={transition:null},vp={ReactCurrentDispatcher:He,ReactCurrentBatchConfig:Fs,ReactCurrentOwner:uo};function Ac(){throw Error("act(...) is not supported in production builds of React.")}ue.Children={map:gs,forEach:function(e,t,n){gs(e,function(){t.apply(this,arguments)},n)},count:function(e){var t=0;return gs(e,function(){t++}),t},toArray:function(e){return gs(e,function(t){return t})||[]},only:function(e){if(!po(e))throw Error("React.Children.only expected to receive a single React element child.");return e}};ue.Component=un;ue.Fragment=ap;ue.Profiler=op;ue.PureComponent=io;ue.StrictMode=lp;ue.Suspense=up;ue.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=vp;ue.act=Ac;ue.cloneElement=function(e,t,n){if(e==null)throw Error("React.cloneElement(...): The argument must be a React element, but you passed "+e+".");var s=Mc({},e.props),a=e.key,l=e.ref,o=e._owner;if(t!=null){if(t.ref!==void 0&&(l=t.ref,o=uo.current),t.key!==void 0&&(a=""+t.key),e.type&&e.type.defaultProps)var i=e.type.defaultProps;for(d in t)Fc.call(t,d)&&!Dc.hasOwnProperty(d)&&(s[d]=t[d]===void 0&&i!==void 0?i[d]:t[d])}var d=arguments.length-2;if(d===1)s.children=n;else if(1<d){i=Array(d);for(var u=0;u<d;u++)i[u]=arguments[u+2];s.children=i}return{$$typeof:is,type:e.type,key:a,ref:l,props:s,_owner:o}};ue.createContext=function(e){return e={$$typeof:cp,_currentValue:e,_currentValue2:e,_threadCount:0,Provider:null,Consumer:null,_defaultValue:null,_globalName:null},e.Provider={$$typeof:ip,_context:e},e.Consumer=e};ue.createElement=Oc;ue.createFactory=function(e){var t=Oc.bind(null,e);return t.type=e,t};ue.createRef=function(){return{current:null}};ue.forwardRef=function(e){return{$$typeof:dp,render:e}};ue.isValidElement=po;ue.lazy=function(e){return{$$typeof:fp,_payload:{_status:-1,_result:e},_init:gp}};ue.memo=function(e,t){return{$$typeof:pp,type:e,compare:t===void 0?null:t}};ue.startTransition=function(e){var t=Fs.transition;Fs.transition={};try{e()}finally{Fs.transition=t}};ue.unstable_act=Ac;ue.useCallback=function(e,t){return He.current.useCallback(e,t)};ue.useContext=function(e){return He.current.useContext(e)};ue.useDebugValue=function(){};ue.useDeferredValue=function(e){return He.current.useDeferredValue(e)};ue.useEffect=function(e,t){return He.current.useEffect(e,t)};ue.useId=function(){return He.current.useId()};ue.useImperativeHandle=function(e,t,n){return He.current.useImperativeHandle(e,t,n)};ue.useInsertionEffect=function(e,t){return He.current.useInsertionEffect(e,t)};ue.useLayoutEffect=function(e,t){return He.current.useLayoutEffect(e,t)};ue.useMemo=function(e,t){return He.current.useMemo(e,t)};ue.useReducer=function(e,t,n){return He.current.useReducer(e,t,n)};ue.useRef=function(e){return He.current.useRef(e)};ue.useState=function(e){return He.current.useState(e)};ue.useSyncExternalStore=function(e,t,n){return He.current.useSyncExternalStore(e,t,n)};ue.useTransition=function(){return He.current.useTransition()};ue.version="18.3.1";Pc.exports=ue;var c=Pc.exports;const yp=np(c);/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var jp=c,bp=Symbol.for("react.element"),wp=Symbol.for("react.fragment"),kp=Object.prototype.hasOwnProperty,Sp=jp.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,Np={key:!0,ref:!0,__self:!0,__source:!0};function $c(e,t,n){var s,a={},l=null,o=null;n!==void 0&&(l=""+n),t.key!==void 0&&(l=""+t.key),t.ref!==void 0&&(o=t.ref);for(s in t)kp.call(t,s)&&!Np.hasOwnProperty(s)&&(a[s]=t[s]);if(e&&e.defaultProps)for(s in t=e.defaultProps,t)a[s]===void 0&&(a[s]=t[s]);return{$$typeof:bp,type:e,key:l,ref:o,props:a,_owner:Sp.current}}va.Fragment=wp;va.jsx=$c;va.jsxs=$c;Tc.exports=va;var r=Tc.exports,fl={},Uc={exports:{}},lt={},Vc={exports:{}},Bc={};/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */(function(e){function t(M,B){var Q=M.length;M.push(B);e:for(;0<Q;){var te=Q-1>>>1,oe=M[te];if(0<a(oe,B))M[te]=B,M[Q]=oe,Q=te;else break e}}function n(M){return M.length===0?null:M[0]}function s(M){if(M.length===0)return null;var B=M[0],Q=M.pop();if(Q!==B){M[0]=Q;e:for(var te=0,oe=M.length,T=oe>>>1;te<T;){var v=2*(te+1)-1,K=M[v],b=v+1,D=M[b];if(0>a(K,Q))b<oe&&0>a(D,K)?(M[te]=D,M[b]=Q,te=b):(M[te]=K,M[v]=Q,te=v);else if(b<oe&&0>a(D,Q))M[te]=D,M[b]=Q,te=b;else break e}}return B}function a(M,B){var Q=M.sortIndex-B.sortIndex;return Q!==0?Q:M.id-B.id}if(typeof performance=="object"&&typeof performance.now=="function"){var l=performance;e.unstable_now=function(){return l.now()}}else{var o=Date,i=o.now();e.unstable_now=function(){return o.now()-i}}var d=[],u=[],y=1,g=null,x=3,k=!1,S=!1,z=!1,R=typeof setTimeout=="function"?setTimeout:null,f=typeof clearTimeout=="function"?clearTimeout:null,p=typeof setImmediate<"u"?setImmediate:null;typeof navigator<"u"&&navigator.scheduling!==void 0&&navigator.scheduling.isInputPending!==void 0&&navigator.scheduling.isInputPending.bind(navigator.scheduling);function m(M){for(var B=n(u);B!==null;){if(B.callback===null)s(u);else if(B.startTime<=M)s(u),B.sortIndex=B.expirationTime,t(d,B);else break;B=n(u)}}function h(M){if(z=!1,m(M),!S)if(n(d)!==null)S=!0,$(j);else{var B=n(u);B!==null&&O(h,B.startTime-M)}}function j(M,B){S=!1,z&&(z=!1,f(I),I=-1),k=!0;var Q=x;try{for(m(B),g=n(d);g!==null&&(!(g.expirationTime>B)||M&&!N());){var te=g.callback;if(typeof te=="function"){g.callback=null,x=g.priorityLevel;var oe=te(g.expirationTime<=B);B=e.unstable_now(),typeof oe=="function"?g.callback=oe:g===n(d)&&s(d),m(B)}else s(d);g=n(d)}if(g!==null)var T=!0;else{var v=n(u);v!==null&&O(h,v.startTime-B),T=!1}return T}finally{g=null,x=Q,k=!1}}var _=!1,P=null,I=-1,G=5,H=-1;function N(){return!(e.unstable_now()-H<G)}function C(){if(P!==null){var M=e.unstable_now();H=M;var B=!0;try{B=P(!0,M)}finally{B?L():(_=!1,P=null)}}else _=!1}var L;if(typeof p=="function")L=function(){p(C)};else if(typeof MessageChannel<"u"){var X=new MessageChannel,A=X.port2;X.port1.onmessage=C,L=function(){A.postMessage(null)}}else L=function(){R(C,0)};function $(M){P=M,_||(_=!0,L())}function O(M,B){I=R(function(){M(e.unstable_now())},B)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(M){M.callback=null},e.unstable_continueExecution=function(){S||k||(S=!0,$(j))},e.unstable_forceFrameRate=function(M){0>M||125<M?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):G=0<M?Math.floor(1e3/M):5},e.unstable_getCurrentPriorityLevel=function(){return x},e.unstable_getFirstCallbackNode=function(){return n(d)},e.unstable_next=function(M){switch(x){case 1:case 2:case 3:var B=3;break;default:B=x}var Q=x;x=B;try{return M()}finally{x=Q}},e.unstable_pauseExecution=function(){},e.unstable_requestPaint=function(){},e.unstable_runWithPriority=function(M,B){switch(M){case 1:case 2:case 3:case 4:case 5:break;default:M=3}var Q=x;x=M;try{return B()}finally{x=Q}},e.unstable_scheduleCallback=function(M,B,Q){var te=e.unstable_now();switch(typeof Q=="object"&&Q!==null?(Q=Q.delay,Q=typeof Q=="number"&&0<Q?te+Q:te):Q=te,M){case 1:var oe=-1;break;case 2:oe=250;break;case 5:oe=1073741823;break;case 4:oe=1e4;break;default:oe=5e3}return oe=Q+oe,M={id:y++,callback:B,priorityLevel:M,startTime:Q,expirationTime:oe,sortIndex:-1},Q>te?(M.sortIndex=Q,t(u,M),n(d)===null&&M===n(u)&&(z?(f(I),I=-1):z=!0,O(h,Q-te))):(M.sortIndex=oe,t(d,M),S||k||(S=!0,$(j))),M},e.unstable_shouldYield=N,e.unstable_wrapCallback=function(M){var B=x;return function(){var Q=x;x=B;try{return M.apply(this,arguments)}finally{x=Q}}}})(Bc);Vc.exports=Bc;var Cp=Vc.exports;/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var _p=c,at=Cp;function U(e){for(var t="https://reactjs.org/docs/error-decoder.html?invariant="+e,n=1;n<arguments.length;n++)t+="&args[]="+encodeURIComponent(arguments[n]);return"Minified React error #"+e+"; visit "+t+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}var Wc=new Set,Bn={};function Mr(e,t){rn(e,t),rn(e+"Capture",t)}function rn(e,t){for(Bn[e]=t,e=0;e<t.length;e++)Wc.add(t[e])}var Vt=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),ml=Object.prototype.hasOwnProperty,Ep=/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,ui={},pi={};function zp(e){return ml.call(pi,e)?!0:ml.call(ui,e)?!1:Ep.test(e)?pi[e]=!0:(ui[e]=!0,!1)}function Tp(e,t,n,s){if(n!==null&&n.type===0)return!1;switch(typeof t){case"function":case"symbol":return!0;case"boolean":return s?!1:n!==null?!n.acceptsBooleans:(e=e.toLowerCase().slice(0,5),e!=="data-"&&e!=="aria-");default:return!1}}function Pp(e,t,n,s){if(t===null||typeof t>"u"||Tp(e,t,n,s))return!0;if(s)return!1;if(n!==null)switch(n.type){case 3:return!t;case 4:return t===!1;case 5:return isNaN(t);case 6:return isNaN(t)||1>t}return!1}function Qe(e,t,n,s,a,l,o){this.acceptsBooleans=t===2||t===3||t===4,this.attributeName=s,this.attributeNamespace=a,this.mustUseProperty=n,this.propertyName=e,this.type=t,this.sanitizeURL=l,this.removeEmptyString=o}var De={};"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e){De[e]=new Qe(e,0,!1,e,null,!1,!1)});[["acceptCharset","accept-charset"],["className","class"],["htmlFor","for"],["httpEquiv","http-equiv"]].forEach(function(e){var t=e[0];De[t]=new Qe(t,1,!1,e[1],null,!1,!1)});["contentEditable","draggable","spellCheck","value"].forEach(function(e){De[e]=new Qe(e,2,!1,e.toLowerCase(),null,!1,!1)});["autoReverse","externalResourcesRequired","focusable","preserveAlpha"].forEach(function(e){De[e]=new Qe(e,2,!1,e,null,!1,!1)});"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e){De[e]=new Qe(e,3,!1,e.toLowerCase(),null,!1,!1)});["checked","multiple","muted","selected"].forEach(function(e){De[e]=new Qe(e,3,!0,e,null,!1,!1)});["capture","download"].forEach(function(e){De[e]=new Qe(e,4,!1,e,null,!1,!1)});["cols","rows","size","span"].forEach(function(e){De[e]=new Qe(e,6,!1,e,null,!1,!1)});["rowSpan","start"].forEach(function(e){De[e]=new Qe(e,5,!1,e.toLowerCase(),null,!1,!1)});var fo=/[\-:]([a-z])/g;function mo(e){return e[1].toUpperCase()}"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e){var t=e.replace(fo,mo);De[t]=new Qe(t,1,!1,e,null,!1,!1)});"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e){var t=e.replace(fo,mo);De[t]=new Qe(t,1,!1,e,"http://www.w3.org/1999/xlink",!1,!1)});["xml:base","xml:lang","xml:space"].forEach(function(e){var t=e.replace(fo,mo);De[t]=new Qe(t,1,!1,e,"http://www.w3.org/XML/1998/namespace",!1,!1)});["tabIndex","crossOrigin"].forEach(function(e){De[e]=new Qe(e,1,!1,e.toLowerCase(),null,!1,!1)});De.xlinkHref=new Qe("xlinkHref",1,!1,"xlink:href","http://www.w3.org/1999/xlink",!0,!1);["src","href","action","formAction"].forEach(function(e){De[e]=new Qe(e,1,!1,e.toLowerCase(),null,!0,!0)});function ho(e,t,n,s){var a=De.hasOwnProperty(t)?De[t]:null;(a!==null?a.type!==0:s||!(2<t.length)||t[0]!=="o"&&t[0]!=="O"||t[1]!=="n"&&t[1]!=="N")&&(Pp(t,n,a,s)&&(n=null),s||a===null?zp(t)&&(n===null?e.removeAttribute(t):e.setAttribute(t,""+n)):a.mustUseProperty?e[a.propertyName]=n===null?a.type===3?!1:"":n:(t=a.attributeName,s=a.attributeNamespace,n===null?e.removeAttribute(t):(a=a.type,n=a===3||a===4&&n===!0?"":""+n,s?e.setAttributeNS(s,t,n):e.setAttribute(t,n))))}var Qt=_p.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,vs=Symbol.for("react.element"),Or=Symbol.for("react.portal"),Ar=Symbol.for("react.fragment"),xo=Symbol.for("react.strict_mode"),hl=Symbol.for("react.profiler"),Gc=Symbol.for("react.provider"),Hc=Symbol.for("react.context"),go=Symbol.for("react.forward_ref"),xl=Symbol.for("react.suspense"),gl=Symbol.for("react.suspense_list"),vo=Symbol.for("react.memo"),qt=Symbol.for("react.lazy"),Qc=Symbol.for("react.offscreen"),fi=Symbol.iterator;function jn(e){return e===null||typeof e!="object"?null:(e=fi&&e[fi]||e["@@iterator"],typeof e=="function"?e:null)}var Ne=Object.assign,Aa;function Tn(e){if(Aa===void 0)try{throw Error()}catch(n){var t=n.stack.trim().match(/\n( *(at )?)/);Aa=t&&t[1]||""}return`
`+Aa+e}var $a=!1;function Ua(e,t){if(!e||$a)return"";$a=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{if(t)if(t=function(){throw Error()},Object.defineProperty(t.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(t,[])}catch(u){var s=u}Reflect.construct(e,[],t)}else{try{t.call()}catch(u){s=u}e.call(t.prototype)}else{try{throw Error()}catch(u){s=u}e()}}catch(u){if(u&&s&&typeof u.stack=="string"){for(var a=u.stack.split(`
`),l=s.stack.split(`
`),o=a.length-1,i=l.length-1;1<=o&&0<=i&&a[o]!==l[i];)i--;for(;1<=o&&0<=i;o--,i--)if(a[o]!==l[i]){if(o!==1||i!==1)do if(o--,i--,0>i||a[o]!==l[i]){var d=`
`+a[o].replace(" at new "," at ");return e.displayName&&d.includes("<anonymous>")&&(d=d.replace("<anonymous>",e.displayName)),d}while(1<=o&&0<=i);break}}}finally{$a=!1,Error.prepareStackTrace=n}return(e=e?e.displayName||e.name:"")?Tn(e):""}function Ip(e){switch(e.tag){case 5:return Tn(e.type);case 16:return Tn("Lazy");case 13:return Tn("Suspense");case 19:return Tn("SuspenseList");case 0:case 2:case 15:return e=Ua(e.type,!1),e;case 11:return e=Ua(e.type.render,!1),e;case 1:return e=Ua(e.type,!0),e;default:return""}}function vl(e){if(e==null)return null;if(typeof e=="function")return e.displayName||e.name||null;if(typeof e=="string")return e;switch(e){case Ar:return"Fragment";case Or:return"Portal";case hl:return"Profiler";case xo:return"StrictMode";case xl:return"Suspense";case gl:return"SuspenseList"}if(typeof e=="object")switch(e.$$typeof){case Hc:return(e.displayName||"Context")+".Consumer";case Gc:return(e._context.displayName||"Context")+".Provider";case go:var t=e.render;return e=e.displayName,e||(e=t.displayName||t.name||"",e=e!==""?"ForwardRef("+e+")":"ForwardRef"),e;case vo:return t=e.displayName||null,t!==null?t:vl(e.type)||"Memo";case qt:t=e._payload,e=e._init;try{return vl(e(t))}catch{}}return null}function Mp(e){var t=e.type;switch(e.tag){case 24:return"Cache";case 9:return(t.displayName||"Context")+".Consumer";case 10:return(t._context.displayName||"Context")+".Provider";case 18:return"DehydratedFragment";case 11:return e=t.render,e=e.displayName||e.name||"",t.displayName||(e!==""?"ForwardRef("+e+")":"ForwardRef");case 7:return"Fragment";case 5:return t;case 4:return"Portal";case 3:return"Root";case 6:return"Text";case 16:return vl(t);case 8:return t===xo?"StrictMode":"Mode";case 22:return"Offscreen";case 12:return"Profiler";case 21:return"Scope";case 13:return"Suspense";case 19:return"SuspenseList";case 25:return"TracingMarker";case 1:case 0:case 17:case 2:case 14:case 15:if(typeof t=="function")return t.displayName||t.name||null;if(typeof t=="string")return t}return null}function ur(e){switch(typeof e){case"boolean":case"number":case"string":case"undefined":return e;case"object":return e;default:return""}}function Xc(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()==="input"&&(t==="checkbox"||t==="radio")}function Rp(e){var t=Xc(e)?"checked":"value",n=Object.getOwnPropertyDescriptor(e.constructor.prototype,t),s=""+e[t];if(!e.hasOwnProperty(t)&&typeof n<"u"&&typeof n.get=="function"&&typeof n.set=="function"){var a=n.get,l=n.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return a.call(this)},set:function(o){s=""+o,l.call(this,o)}}),Object.defineProperty(e,t,{enumerable:n.enumerable}),{getValue:function(){return s},setValue:function(o){s=""+o},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function ys(e){e._valueTracker||(e._valueTracker=Rp(e))}function Yc(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),s="";return e&&(s=Xc(e)?e.checked?"true":"false":e.value),e=s,e!==n?(t.setValue(e),!0):!1}function Qs(e){if(e=e||(typeof document<"u"?document:void 0),typeof e>"u")return null;try{return e.activeElement||e.body}catch{return e.body}}function yl(e,t){var n=t.checked;return Ne({},t,{defaultChecked:void 0,defaultValue:void 0,value:void 0,checked:n??e._wrapperState.initialChecked})}function mi(e,t){var n=t.defaultValue==null?"":t.defaultValue,s=t.checked!=null?t.checked:t.defaultChecked;n=ur(t.value!=null?t.value:n),e._wrapperState={initialChecked:s,initialValue:n,controlled:t.type==="checkbox"||t.type==="radio"?t.checked!=null:t.value!=null}}function Kc(e,t){t=t.checked,t!=null&&ho(e,"checked",t,!1)}function jl(e,t){Kc(e,t);var n=ur(t.value),s=t.type;if(n!=null)s==="number"?(n===0&&e.value===""||e.value!=n)&&(e.value=""+n):e.value!==""+n&&(e.value=""+n);else if(s==="submit"||s==="reset"){e.removeAttribute("value");return}t.hasOwnProperty("value")?bl(e,t.type,n):t.hasOwnProperty("defaultValue")&&bl(e,t.type,ur(t.defaultValue)),t.checked==null&&t.defaultChecked!=null&&(e.defaultChecked=!!t.defaultChecked)}function hi(e,t,n){if(t.hasOwnProperty("value")||t.hasOwnProperty("defaultValue")){var s=t.type;if(!(s!=="submit"&&s!=="reset"||t.value!==void 0&&t.value!==null))return;t=""+e._wrapperState.initialValue,n||t===e.value||(e.value=t),e.defaultValue=t}n=e.name,n!==""&&(e.name=""),e.defaultChecked=!!e._wrapperState.initialChecked,n!==""&&(e.name=n)}function bl(e,t,n){(t!=="number"||Qs(e.ownerDocument)!==e)&&(n==null?e.defaultValue=""+e._wrapperState.initialValue:e.defaultValue!==""+n&&(e.defaultValue=""+n))}var Pn=Array.isArray;function Kr(e,t,n,s){if(e=e.options,t){t={};for(var a=0;a<n.length;a++)t["$"+n[a]]=!0;for(n=0;n<e.length;n++)a=t.hasOwnProperty("$"+e[n].value),e[n].selected!==a&&(e[n].selected=a),a&&s&&(e[n].defaultSelected=!0)}else{for(n=""+ur(n),t=null,a=0;a<e.length;a++){if(e[a].value===n){e[a].selected=!0,s&&(e[a].defaultSelected=!0);return}t!==null||e[a].disabled||(t=e[a])}t!==null&&(t.selected=!0)}}function wl(e,t){if(t.dangerouslySetInnerHTML!=null)throw Error(U(91));return Ne({},t,{value:void 0,defaultValue:void 0,children:""+e._wrapperState.initialValue})}function xi(e,t){var n=t.value;if(n==null){if(n=t.children,t=t.defaultValue,n!=null){if(t!=null)throw Error(U(92));if(Pn(n)){if(1<n.length)throw Error(U(93));n=n[0]}t=n}t==null&&(t=""),n=t}e._wrapperState={initialValue:ur(n)}}function qc(e,t){var n=ur(t.value),s=ur(t.defaultValue);n!=null&&(n=""+n,n!==e.value&&(e.value=n),t.defaultValue==null&&e.defaultValue!==n&&(e.defaultValue=n)),s!=null&&(e.defaultValue=""+s)}function gi(e){var t=e.textContent;t===e._wrapperState.initialValue&&t!==""&&t!==null&&(e.value=t)}function Jc(e){switch(e){case"svg":return"http://www.w3.org/2000/svg";case"math":return"http://www.w3.org/1998/Math/MathML";default:return"http://www.w3.org/1999/xhtml"}}function kl(e,t){return e==null||e==="http://www.w3.org/1999/xhtml"?Jc(t):e==="http://www.w3.org/2000/svg"&&t==="foreignObject"?"http://www.w3.org/1999/xhtml":e}var js,Zc=function(e){return typeof MSApp<"u"&&MSApp.execUnsafeLocalFunction?function(t,n,s,a){MSApp.execUnsafeLocalFunction(function(){return e(t,n,s,a)})}:e}(function(e,t){if(e.namespaceURI!=="http://www.w3.org/2000/svg"||"innerHTML"in e)e.innerHTML=t;else{for(js=js||document.createElement("div"),js.innerHTML="<svg>"+t.valueOf().toString()+"</svg>",t=js.firstChild;e.firstChild;)e.removeChild(e.firstChild);for(;t.firstChild;)e.appendChild(t.firstChild)}});function Wn(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var Rn={animationIterationCount:!0,aspectRatio:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridArea:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},Lp=["Webkit","ms","Moz","O"];Object.keys(Rn).forEach(function(e){Lp.forEach(function(t){t=t+e.charAt(0).toUpperCase()+e.substring(1),Rn[t]=Rn[e]})});function ed(e,t,n){return t==null||typeof t=="boolean"||t===""?"":n||typeof t!="number"||t===0||Rn.hasOwnProperty(e)&&Rn[e]?(""+t).trim():t+"px"}function td(e,t){e=e.style;for(var n in t)if(t.hasOwnProperty(n)){var s=n.indexOf("--")===0,a=ed(n,t[n],s);n==="float"&&(n="cssFloat"),s?e.setProperty(n,a):e[n]=a}}var Fp=Ne({menuitem:!0},{area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0});function Sl(e,t){if(t){if(Fp[e]&&(t.children!=null||t.dangerouslySetInnerHTML!=null))throw Error(U(137,e));if(t.dangerouslySetInnerHTML!=null){if(t.children!=null)throw Error(U(60));if(typeof t.dangerouslySetInnerHTML!="object"||!("__html"in t.dangerouslySetInnerHTML))throw Error(U(61))}if(t.style!=null&&typeof t.style!="object")throw Error(U(62))}}function Nl(e,t){if(e.indexOf("-")===-1)return typeof t.is=="string";switch(e){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var Cl=null;function yo(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var _l=null,qr=null,Jr=null;function vi(e){if(e=us(e)){if(typeof _l!="function")throw Error(U(280));var t=e.stateNode;t&&(t=ka(t),_l(e.stateNode,e.type,t))}}function rd(e){qr?Jr?Jr.push(e):Jr=[e]:qr=e}function nd(){if(qr){var e=qr,t=Jr;if(Jr=qr=null,vi(e),t)for(e=0;e<t.length;e++)vi(t[e])}}function sd(e,t){return e(t)}function ad(){}var Va=!1;function ld(e,t,n){if(Va)return e(t,n);Va=!0;try{return sd(e,t,n)}finally{Va=!1,(qr!==null||Jr!==null)&&(ad(),nd())}}function Gn(e,t){var n=e.stateNode;if(n===null)return null;var s=ka(n);if(s===null)return null;n=s[t];e:switch(t){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(s=!s.disabled)||(e=e.type,s=!(e==="button"||e==="input"||e==="select"||e==="textarea")),e=!s;break e;default:e=!1}if(e)return null;if(n&&typeof n!="function")throw Error(U(231,t,typeof n));return n}var El=!1;if(Vt)try{var bn={};Object.defineProperty(bn,"passive",{get:function(){El=!0}}),window.addEventListener("test",bn,bn),window.removeEventListener("test",bn,bn)}catch{El=!1}function Dp(e,t,n,s,a,l,o,i,d){var u=Array.prototype.slice.call(arguments,3);try{t.apply(n,u)}catch(y){this.onError(y)}}var Ln=!1,Xs=null,Ys=!1,zl=null,Op={onError:function(e){Ln=!0,Xs=e}};function Ap(e,t,n,s,a,l,o,i,d){Ln=!1,Xs=null,Dp.apply(Op,arguments)}function $p(e,t,n,s,a,l,o,i,d){if(Ap.apply(this,arguments),Ln){if(Ln){var u=Xs;Ln=!1,Xs=null}else throw Error(U(198));Ys||(Ys=!0,zl=u)}}function Rr(e){var t=e,n=e;if(e.alternate)for(;t.return;)t=t.return;else{e=t;do t=e,t.flags&4098&&(n=t.return),e=t.return;while(e)}return t.tag===3?n:null}function od(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function yi(e){if(Rr(e)!==e)throw Error(U(188))}function Up(e){var t=e.alternate;if(!t){if(t=Rr(e),t===null)throw Error(U(188));return t!==e?null:e}for(var n=e,s=t;;){var a=n.return;if(a===null)break;var l=a.alternate;if(l===null){if(s=a.return,s!==null){n=s;continue}break}if(a.child===l.child){for(l=a.child;l;){if(l===n)return yi(a),e;if(l===s)return yi(a),t;l=l.sibling}throw Error(U(188))}if(n.return!==s.return)n=a,s=l;else{for(var o=!1,i=a.child;i;){if(i===n){o=!0,n=a,s=l;break}if(i===s){o=!0,s=a,n=l;break}i=i.sibling}if(!o){for(i=l.child;i;){if(i===n){o=!0,n=l,s=a;break}if(i===s){o=!0,s=l,n=a;break}i=i.sibling}if(!o)throw Error(U(189))}}if(n.alternate!==s)throw Error(U(190))}if(n.tag!==3)throw Error(U(188));return n.stateNode.current===n?e:t}function id(e){return e=Up(e),e!==null?cd(e):null}function cd(e){if(e.tag===5||e.tag===6)return e;for(e=e.child;e!==null;){var t=cd(e);if(t!==null)return t;e=e.sibling}return null}var dd=at.unstable_scheduleCallback,ji=at.unstable_cancelCallback,Vp=at.unstable_shouldYield,Bp=at.unstable_requestPaint,_e=at.unstable_now,Wp=at.unstable_getCurrentPriorityLevel,jo=at.unstable_ImmediatePriority,ud=at.unstable_UserBlockingPriority,Ks=at.unstable_NormalPriority,Gp=at.unstable_LowPriority,pd=at.unstable_IdlePriority,ya=null,Tt=null;function Hp(e){if(Tt&&typeof Tt.onCommitFiberRoot=="function")try{Tt.onCommitFiberRoot(ya,e,void 0,(e.current.flags&128)===128)}catch{}}var bt=Math.clz32?Math.clz32:Yp,Qp=Math.log,Xp=Math.LN2;function Yp(e){return e>>>=0,e===0?32:31-(Qp(e)/Xp|0)|0}var bs=64,ws=4194304;function In(e){switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e&4194240;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return e&130023424;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 1073741824;default:return e}}function qs(e,t){var n=e.pendingLanes;if(n===0)return 0;var s=0,a=e.suspendedLanes,l=e.pingedLanes,o=n&268435455;if(o!==0){var i=o&~a;i!==0?s=In(i):(l&=o,l!==0&&(s=In(l)))}else o=n&~a,o!==0?s=In(o):l!==0&&(s=In(l));if(s===0)return 0;if(t!==0&&t!==s&&!(t&a)&&(a=s&-s,l=t&-t,a>=l||a===16&&(l&4194240)!==0))return t;if(s&4&&(s|=n&16),t=e.entangledLanes,t!==0)for(e=e.entanglements,t&=s;0<t;)n=31-bt(t),a=1<<n,s|=e[n],t&=~a;return s}function Kp(e,t){switch(e){case 1:case 2:case 4:return t+250;case 8:case 16:case 32:case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return-1;case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function qp(e,t){for(var n=e.suspendedLanes,s=e.pingedLanes,a=e.expirationTimes,l=e.pendingLanes;0<l;){var o=31-bt(l),i=1<<o,d=a[o];d===-1?(!(i&n)||i&s)&&(a[o]=Kp(i,t)):d<=t&&(e.expiredLanes|=i),l&=~i}}function Tl(e){return e=e.pendingLanes&-1073741825,e!==0?e:e&1073741824?1073741824:0}function fd(){var e=bs;return bs<<=1,!(bs&4194240)&&(bs=64),e}function Ba(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function cs(e,t,n){e.pendingLanes|=t,t!==536870912&&(e.suspendedLanes=0,e.pingedLanes=0),e=e.eventTimes,t=31-bt(t),e[t]=n}function Jp(e,t){var n=e.pendingLanes&~t;e.pendingLanes=t,e.suspendedLanes=0,e.pingedLanes=0,e.expiredLanes&=t,e.mutableReadLanes&=t,e.entangledLanes&=t,t=e.entanglements;var s=e.eventTimes;for(e=e.expirationTimes;0<n;){var a=31-bt(n),l=1<<a;t[a]=0,s[a]=-1,e[a]=-1,n&=~l}}function bo(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var s=31-bt(n),a=1<<s;a&t|e[s]&t&&(e[s]|=t),n&=~a}}var ge=0;function md(e){return e&=-e,1<e?4<e?e&268435455?16:536870912:4:1}var hd,wo,xd,gd,vd,Pl=!1,ks=[],nr=null,sr=null,ar=null,Hn=new Map,Qn=new Map,Zt=[],Zp="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");function bi(e,t){switch(e){case"focusin":case"focusout":nr=null;break;case"dragenter":case"dragleave":sr=null;break;case"mouseover":case"mouseout":ar=null;break;case"pointerover":case"pointerout":Hn.delete(t.pointerId);break;case"gotpointercapture":case"lostpointercapture":Qn.delete(t.pointerId)}}function wn(e,t,n,s,a,l){return e===null||e.nativeEvent!==l?(e={blockedOn:t,domEventName:n,eventSystemFlags:s,nativeEvent:l,targetContainers:[a]},t!==null&&(t=us(t),t!==null&&wo(t)),e):(e.eventSystemFlags|=s,t=e.targetContainers,a!==null&&t.indexOf(a)===-1&&t.push(a),e)}function ef(e,t,n,s,a){switch(t){case"focusin":return nr=wn(nr,e,t,n,s,a),!0;case"dragenter":return sr=wn(sr,e,t,n,s,a),!0;case"mouseover":return ar=wn(ar,e,t,n,s,a),!0;case"pointerover":var l=a.pointerId;return Hn.set(l,wn(Hn.get(l)||null,e,t,n,s,a)),!0;case"gotpointercapture":return l=a.pointerId,Qn.set(l,wn(Qn.get(l)||null,e,t,n,s,a)),!0}return!1}function yd(e){var t=kr(e.target);if(t!==null){var n=Rr(t);if(n!==null){if(t=n.tag,t===13){if(t=od(n),t!==null){e.blockedOn=t,vd(e.priority,function(){xd(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function Ds(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=Il(e.domEventName,e.eventSystemFlags,t[0],e.nativeEvent);if(n===null){n=e.nativeEvent;var s=new n.constructor(n.type,n);Cl=s,n.target.dispatchEvent(s),Cl=null}else return t=us(n),t!==null&&wo(t),e.blockedOn=n,!1;t.shift()}return!0}function wi(e,t,n){Ds(e)&&n.delete(t)}function tf(){Pl=!1,nr!==null&&Ds(nr)&&(nr=null),sr!==null&&Ds(sr)&&(sr=null),ar!==null&&Ds(ar)&&(ar=null),Hn.forEach(wi),Qn.forEach(wi)}function kn(e,t){e.blockedOn===t&&(e.blockedOn=null,Pl||(Pl=!0,at.unstable_scheduleCallback(at.unstable_NormalPriority,tf)))}function Xn(e){function t(a){return kn(a,e)}if(0<ks.length){kn(ks[0],e);for(var n=1;n<ks.length;n++){var s=ks[n];s.blockedOn===e&&(s.blockedOn=null)}}for(nr!==null&&kn(nr,e),sr!==null&&kn(sr,e),ar!==null&&kn(ar,e),Hn.forEach(t),Qn.forEach(t),n=0;n<Zt.length;n++)s=Zt[n],s.blockedOn===e&&(s.blockedOn=null);for(;0<Zt.length&&(n=Zt[0],n.blockedOn===null);)yd(n),n.blockedOn===null&&Zt.shift()}var Zr=Qt.ReactCurrentBatchConfig,Js=!0;function rf(e,t,n,s){var a=ge,l=Zr.transition;Zr.transition=null;try{ge=1,ko(e,t,n,s)}finally{ge=a,Zr.transition=l}}function nf(e,t,n,s){var a=ge,l=Zr.transition;Zr.transition=null;try{ge=4,ko(e,t,n,s)}finally{ge=a,Zr.transition=l}}function ko(e,t,n,s){if(Js){var a=Il(e,t,n,s);if(a===null)Za(e,t,s,Zs,n),bi(e,s);else if(ef(a,e,t,n,s))s.stopPropagation();else if(bi(e,s),t&4&&-1<Zp.indexOf(e)){for(;a!==null;){var l=us(a);if(l!==null&&hd(l),l=Il(e,t,n,s),l===null&&Za(e,t,s,Zs,n),l===a)break;a=l}a!==null&&s.stopPropagation()}else Za(e,t,s,null,n)}}var Zs=null;function Il(e,t,n,s){if(Zs=null,e=yo(s),e=kr(e),e!==null)if(t=Rr(e),t===null)e=null;else if(n=t.tag,n===13){if(e=od(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null);return Zs=e,null}function jd(e){switch(e){case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 1;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"toggle":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 4;case"message":switch(Wp()){case jo:return 1;case ud:return 4;case Ks:case Gp:return 16;case pd:return 536870912;default:return 16}default:return 16}}var tr=null,So=null,Os=null;function bd(){if(Os)return Os;var e,t=So,n=t.length,s,a="value"in tr?tr.value:tr.textContent,l=a.length;for(e=0;e<n&&t[e]===a[e];e++);var o=n-e;for(s=1;s<=o&&t[n-s]===a[l-s];s++);return Os=a.slice(e,1<s?1-s:void 0)}function As(e){var t=e.keyCode;return"charCode"in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function Ss(){return!0}function ki(){return!1}function ot(e){function t(n,s,a,l,o){this._reactName=n,this._targetInst=a,this.type=s,this.nativeEvent=l,this.target=o,this.currentTarget=null;for(var i in e)e.hasOwnProperty(i)&&(n=e[i],this[i]=n?n(l):l[i]);return this.isDefaultPrevented=(l.defaultPrevented!=null?l.defaultPrevented:l.returnValue===!1)?Ss:ki,this.isPropagationStopped=ki,this}return Ne(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=Ss)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=Ss)},persist:function(){},isPersistent:Ss}),t}var pn={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},No=ot(pn),ds=Ne({},pn,{view:0,detail:0}),sf=ot(ds),Wa,Ga,Sn,ja=Ne({},ds,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:Co,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return"movementX"in e?e.movementX:(e!==Sn&&(Sn&&e.type==="mousemove"?(Wa=e.screenX-Sn.screenX,Ga=e.screenY-Sn.screenY):Ga=Wa=0,Sn=e),Wa)},movementY:function(e){return"movementY"in e?e.movementY:Ga}}),Si=ot(ja),af=Ne({},ja,{dataTransfer:0}),lf=ot(af),of=Ne({},ds,{relatedTarget:0}),Ha=ot(of),cf=Ne({},pn,{animationName:0,elapsedTime:0,pseudoElement:0}),df=ot(cf),uf=Ne({},pn,{clipboardData:function(e){return"clipboardData"in e?e.clipboardData:window.clipboardData}}),pf=ot(uf),ff=Ne({},pn,{data:0}),Ni=ot(ff),mf={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},hf={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},xf={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function gf(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=xf[e])?!!t[e]:!1}function Co(){return gf}var vf=Ne({},ds,{key:function(e){if(e.key){var t=mf[e.key]||e.key;if(t!=="Unidentified")return t}return e.type==="keypress"?(e=As(e),e===13?"Enter":String.fromCharCode(e)):e.type==="keydown"||e.type==="keyup"?hf[e.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:Co,charCode:function(e){return e.type==="keypress"?As(e):0},keyCode:function(e){return e.type==="keydown"||e.type==="keyup"?e.keyCode:0},which:function(e){return e.type==="keypress"?As(e):e.type==="keydown"||e.type==="keyup"?e.keyCode:0}}),yf=ot(vf),jf=Ne({},ja,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),Ci=ot(jf),bf=Ne({},ds,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:Co}),wf=ot(bf),kf=Ne({},pn,{propertyName:0,elapsedTime:0,pseudoElement:0}),Sf=ot(kf),Nf=Ne({},ja,{deltaX:function(e){return"deltaX"in e?e.deltaX:"wheelDeltaX"in e?-e.wheelDeltaX:0},deltaY:function(e){return"deltaY"in e?e.deltaY:"wheelDeltaY"in e?-e.wheelDeltaY:"wheelDelta"in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0}),Cf=ot(Nf),_f=[9,13,27,32],_o=Vt&&"CompositionEvent"in window,Fn=null;Vt&&"documentMode"in document&&(Fn=document.documentMode);var Ef=Vt&&"TextEvent"in window&&!Fn,wd=Vt&&(!_o||Fn&&8<Fn&&11>=Fn),_i=" ",Ei=!1;function kd(e,t){switch(e){case"keyup":return _f.indexOf(t.keyCode)!==-1;case"keydown":return t.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function Sd(e){return e=e.detail,typeof e=="object"&&"data"in e?e.data:null}var $r=!1;function zf(e,t){switch(e){case"compositionend":return Sd(t);case"keypress":return t.which!==32?null:(Ei=!0,_i);case"textInput":return e=t.data,e===_i&&Ei?null:e;default:return null}}function Tf(e,t){if($r)return e==="compositionend"||!_o&&kd(e,t)?(e=bd(),Os=So=tr=null,$r=!1,e):null;switch(e){case"paste":return null;case"keypress":if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case"compositionend":return wd&&t.locale!=="ko"?null:t.data;default:return null}}var Pf={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function zi(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t==="input"?!!Pf[e.type]:t==="textarea"}function Nd(e,t,n,s){rd(s),t=ea(t,"onChange"),0<t.length&&(n=new No("onChange","change",null,n,s),e.push({event:n,listeners:t}))}var Dn=null,Yn=null;function If(e){Fd(e,0)}function ba(e){var t=Br(e);if(Yc(t))return e}function Mf(e,t){if(e==="change")return t}var Cd=!1;if(Vt){var Qa;if(Vt){var Xa="oninput"in document;if(!Xa){var Ti=document.createElement("div");Ti.setAttribute("oninput","return;"),Xa=typeof Ti.oninput=="function"}Qa=Xa}else Qa=!1;Cd=Qa&&(!document.documentMode||9<document.documentMode)}function Pi(){Dn&&(Dn.detachEvent("onpropertychange",_d),Yn=Dn=null)}function _d(e){if(e.propertyName==="value"&&ba(Yn)){var t=[];Nd(t,Yn,e,yo(e)),ld(If,t)}}function Rf(e,t,n){e==="focusin"?(Pi(),Dn=t,Yn=n,Dn.attachEvent("onpropertychange",_d)):e==="focusout"&&Pi()}function Lf(e){if(e==="selectionchange"||e==="keyup"||e==="keydown")return ba(Yn)}function Ff(e,t){if(e==="click")return ba(t)}function Df(e,t){if(e==="input"||e==="change")return ba(t)}function Of(e,t){return e===t&&(e!==0||1/e===1/t)||e!==e&&t!==t}var kt=typeof Object.is=="function"?Object.is:Of;function Kn(e,t){if(kt(e,t))return!0;if(typeof e!="object"||e===null||typeof t!="object"||t===null)return!1;var n=Object.keys(e),s=Object.keys(t);if(n.length!==s.length)return!1;for(s=0;s<n.length;s++){var a=n[s];if(!ml.call(t,a)||!kt(e[a],t[a]))return!1}return!0}function Ii(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function Mi(e,t){var n=Ii(e);e=0;for(var s;n;){if(n.nodeType===3){if(s=e+n.textContent.length,e<=t&&s>=t)return{node:n,offset:t-e};e=s}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=Ii(n)}}function Ed(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?Ed(e,t.parentNode):"contains"in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function zd(){for(var e=window,t=Qs();t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href=="string"}catch{n=!1}if(n)e=t.contentWindow;else break;t=Qs(e.document)}return t}function Eo(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t==="input"&&(e.type==="text"||e.type==="search"||e.type==="tel"||e.type==="url"||e.type==="password")||t==="textarea"||e.contentEditable==="true")}function Af(e){var t=zd(),n=e.focusedElem,s=e.selectionRange;if(t!==n&&n&&n.ownerDocument&&Ed(n.ownerDocument.documentElement,n)){if(s!==null&&Eo(n)){if(t=s.start,e=s.end,e===void 0&&(e=t),"selectionStart"in n)n.selectionStart=t,n.selectionEnd=Math.min(e,n.value.length);else if(e=(t=n.ownerDocument||document)&&t.defaultView||window,e.getSelection){e=e.getSelection();var a=n.textContent.length,l=Math.min(s.start,a);s=s.end===void 0?l:Math.min(s.end,a),!e.extend&&l>s&&(a=s,s=l,l=a),a=Mi(n,l);var o=Mi(n,s);a&&o&&(e.rangeCount!==1||e.anchorNode!==a.node||e.anchorOffset!==a.offset||e.focusNode!==o.node||e.focusOffset!==o.offset)&&(t=t.createRange(),t.setStart(a.node,a.offset),e.removeAllRanges(),l>s?(e.addRange(t),e.extend(o.node,o.offset)):(t.setEnd(o.node,o.offset),e.addRange(t)))}}for(t=[],e=n;e=e.parentNode;)e.nodeType===1&&t.push({element:e,left:e.scrollLeft,top:e.scrollTop});for(typeof n.focus=="function"&&n.focus(),n=0;n<t.length;n++)e=t[n],e.element.scrollLeft=e.left,e.element.scrollTop=e.top}}var $f=Vt&&"documentMode"in document&&11>=document.documentMode,Ur=null,Ml=null,On=null,Rl=!1;function Ri(e,t,n){var s=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;Rl||Ur==null||Ur!==Qs(s)||(s=Ur,"selectionStart"in s&&Eo(s)?s={start:s.selectionStart,end:s.selectionEnd}:(s=(s.ownerDocument&&s.ownerDocument.defaultView||window).getSelection(),s={anchorNode:s.anchorNode,anchorOffset:s.anchorOffset,focusNode:s.focusNode,focusOffset:s.focusOffset}),On&&Kn(On,s)||(On=s,s=ea(Ml,"onSelect"),0<s.length&&(t=new No("onSelect","select",null,t,n),e.push({event:t,listeners:s}),t.target=Ur)))}function Ns(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n["Webkit"+e]="webkit"+t,n["Moz"+e]="moz"+t,n}var Vr={animationend:Ns("Animation","AnimationEnd"),animationiteration:Ns("Animation","AnimationIteration"),animationstart:Ns("Animation","AnimationStart"),transitionend:Ns("Transition","TransitionEnd")},Ya={},Td={};Vt&&(Td=document.createElement("div").style,"AnimationEvent"in window||(delete Vr.animationend.animation,delete Vr.animationiteration.animation,delete Vr.animationstart.animation),"TransitionEvent"in window||delete Vr.transitionend.transition);function wa(e){if(Ya[e])return Ya[e];if(!Vr[e])return e;var t=Vr[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in Td)return Ya[e]=t[n];return e}var Pd=wa("animationend"),Id=wa("animationiteration"),Md=wa("animationstart"),Rd=wa("transitionend"),Ld=new Map,Li="abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");function xr(e,t){Ld.set(e,t),Mr(t,[e])}for(var Ka=0;Ka<Li.length;Ka++){var qa=Li[Ka],Uf=qa.toLowerCase(),Vf=qa[0].toUpperCase()+qa.slice(1);xr(Uf,"on"+Vf)}xr(Pd,"onAnimationEnd");xr(Id,"onAnimationIteration");xr(Md,"onAnimationStart");xr("dblclick","onDoubleClick");xr("focusin","onFocus");xr("focusout","onBlur");xr(Rd,"onTransitionEnd");rn("onMouseEnter",["mouseout","mouseover"]);rn("onMouseLeave",["mouseout","mouseover"]);rn("onPointerEnter",["pointerout","pointerover"]);rn("onPointerLeave",["pointerout","pointerover"]);Mr("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));Mr("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));Mr("onBeforeInput",["compositionend","keypress","textInput","paste"]);Mr("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));Mr("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));Mr("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var Mn="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),Bf=new Set("cancel close invalid load scroll toggle".split(" ").concat(Mn));function Fi(e,t,n){var s=e.type||"unknown-event";e.currentTarget=n,$p(s,t,void 0,e),e.currentTarget=null}function Fd(e,t){t=(t&4)!==0;for(var n=0;n<e.length;n++){var s=e[n],a=s.event;s=s.listeners;e:{var l=void 0;if(t)for(var o=s.length-1;0<=o;o--){var i=s[o],d=i.instance,u=i.currentTarget;if(i=i.listener,d!==l&&a.isPropagationStopped())break e;Fi(a,i,u),l=d}else for(o=0;o<s.length;o++){if(i=s[o],d=i.instance,u=i.currentTarget,i=i.listener,d!==l&&a.isPropagationStopped())break e;Fi(a,i,u),l=d}}}if(Ys)throw e=zl,Ys=!1,zl=null,e}function je(e,t){var n=t[Al];n===void 0&&(n=t[Al]=new Set);var s=e+"__bubble";n.has(s)||(Dd(t,e,2,!1),n.add(s))}function Ja(e,t,n){var s=0;t&&(s|=4),Dd(n,e,s,t)}var Cs="_reactListening"+Math.random().toString(36).slice(2);function qn(e){if(!e[Cs]){e[Cs]=!0,Wc.forEach(function(n){n!=="selectionchange"&&(Bf.has(n)||Ja(n,!1,e),Ja(n,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[Cs]||(t[Cs]=!0,Ja("selectionchange",!1,t))}}function Dd(e,t,n,s){switch(jd(t)){case 1:var a=rf;break;case 4:a=nf;break;default:a=ko}n=a.bind(null,t,n,e),a=void 0,!El||t!=="touchstart"&&t!=="touchmove"&&t!=="wheel"||(a=!0),s?a!==void 0?e.addEventListener(t,n,{capture:!0,passive:a}):e.addEventListener(t,n,!0):a!==void 0?e.addEventListener(t,n,{passive:a}):e.addEventListener(t,n,!1)}function Za(e,t,n,s,a){var l=s;if(!(t&1)&&!(t&2)&&s!==null)e:for(;;){if(s===null)return;var o=s.tag;if(o===3||o===4){var i=s.stateNode.containerInfo;if(i===a||i.nodeType===8&&i.parentNode===a)break;if(o===4)for(o=s.return;o!==null;){var d=o.tag;if((d===3||d===4)&&(d=o.stateNode.containerInfo,d===a||d.nodeType===8&&d.parentNode===a))return;o=o.return}for(;i!==null;){if(o=kr(i),o===null)return;if(d=o.tag,d===5||d===6){s=l=o;continue e}i=i.parentNode}}s=s.return}ld(function(){var u=l,y=yo(n),g=[];e:{var x=Ld.get(e);if(x!==void 0){var k=No,S=e;switch(e){case"keypress":if(As(n)===0)break e;case"keydown":case"keyup":k=yf;break;case"focusin":S="focus",k=Ha;break;case"focusout":S="blur",k=Ha;break;case"beforeblur":case"afterblur":k=Ha;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":k=Si;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":k=lf;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":k=wf;break;case Pd:case Id:case Md:k=df;break;case Rd:k=Sf;break;case"scroll":k=sf;break;case"wheel":k=Cf;break;case"copy":case"cut":case"paste":k=pf;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":k=Ci}var z=(t&4)!==0,R=!z&&e==="scroll",f=z?x!==null?x+"Capture":null:x;z=[];for(var p=u,m;p!==null;){m=p;var h=m.stateNode;if(m.tag===5&&h!==null&&(m=h,f!==null&&(h=Gn(p,f),h!=null&&z.push(Jn(p,h,m)))),R)break;p=p.return}0<z.length&&(x=new k(x,S,null,n,y),g.push({event:x,listeners:z}))}}if(!(t&7)){e:{if(x=e==="mouseover"||e==="pointerover",k=e==="mouseout"||e==="pointerout",x&&n!==Cl&&(S=n.relatedTarget||n.fromElement)&&(kr(S)||S[Bt]))break e;if((k||x)&&(x=y.window===y?y:(x=y.ownerDocument)?x.defaultView||x.parentWindow:window,k?(S=n.relatedTarget||n.toElement,k=u,S=S?kr(S):null,S!==null&&(R=Rr(S),S!==R||S.tag!==5&&S.tag!==6)&&(S=null)):(k=null,S=u),k!==S)){if(z=Si,h="onMouseLeave",f="onMouseEnter",p="mouse",(e==="pointerout"||e==="pointerover")&&(z=Ci,h="onPointerLeave",f="onPointerEnter",p="pointer"),R=k==null?x:Br(k),m=S==null?x:Br(S),x=new z(h,p+"leave",k,n,y),x.target=R,x.relatedTarget=m,h=null,kr(y)===u&&(z=new z(f,p+"enter",S,n,y),z.target=m,z.relatedTarget=R,h=z),R=h,k&&S)t:{for(z=k,f=S,p=0,m=z;m;m=Dr(m))p++;for(m=0,h=f;h;h=Dr(h))m++;for(;0<p-m;)z=Dr(z),p--;for(;0<m-p;)f=Dr(f),m--;for(;p--;){if(z===f||f!==null&&z===f.alternate)break t;z=Dr(z),f=Dr(f)}z=null}else z=null;k!==null&&Di(g,x,k,z,!1),S!==null&&R!==null&&Di(g,R,S,z,!0)}}e:{if(x=u?Br(u):window,k=x.nodeName&&x.nodeName.toLowerCase(),k==="select"||k==="input"&&x.type==="file")var j=Mf;else if(zi(x))if(Cd)j=Df;else{j=Lf;var _=Rf}else(k=x.nodeName)&&k.toLowerCase()==="input"&&(x.type==="checkbox"||x.type==="radio")&&(j=Ff);if(j&&(j=j(e,u))){Nd(g,j,n,y);break e}_&&_(e,x,u),e==="focusout"&&(_=x._wrapperState)&&_.controlled&&x.type==="number"&&bl(x,"number",x.value)}switch(_=u?Br(u):window,e){case"focusin":(zi(_)||_.contentEditable==="true")&&(Ur=_,Ml=u,On=null);break;case"focusout":On=Ml=Ur=null;break;case"mousedown":Rl=!0;break;case"contextmenu":case"mouseup":case"dragend":Rl=!1,Ri(g,n,y);break;case"selectionchange":if($f)break;case"keydown":case"keyup":Ri(g,n,y)}var P;if(_o)e:{switch(e){case"compositionstart":var I="onCompositionStart";break e;case"compositionend":I="onCompositionEnd";break e;case"compositionupdate":I="onCompositionUpdate";break e}I=void 0}else $r?kd(e,n)&&(I="onCompositionEnd"):e==="keydown"&&n.keyCode===229&&(I="onCompositionStart");I&&(wd&&n.locale!=="ko"&&($r||I!=="onCompositionStart"?I==="onCompositionEnd"&&$r&&(P=bd()):(tr=y,So="value"in tr?tr.value:tr.textContent,$r=!0)),_=ea(u,I),0<_.length&&(I=new Ni(I,e,null,n,y),g.push({event:I,listeners:_}),P?I.data=P:(P=Sd(n),P!==null&&(I.data=P)))),(P=Ef?zf(e,n):Tf(e,n))&&(u=ea(u,"onBeforeInput"),0<u.length&&(y=new Ni("onBeforeInput","beforeinput",null,n,y),g.push({event:y,listeners:u}),y.data=P))}Fd(g,t)})}function Jn(e,t,n){return{instance:e,listener:t,currentTarget:n}}function ea(e,t){for(var n=t+"Capture",s=[];e!==null;){var a=e,l=a.stateNode;a.tag===5&&l!==null&&(a=l,l=Gn(e,n),l!=null&&s.unshift(Jn(e,l,a)),l=Gn(e,t),l!=null&&s.push(Jn(e,l,a))),e=e.return}return s}function Dr(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5);return e||null}function Di(e,t,n,s,a){for(var l=t._reactName,o=[];n!==null&&n!==s;){var i=n,d=i.alternate,u=i.stateNode;if(d!==null&&d===s)break;i.tag===5&&u!==null&&(i=u,a?(d=Gn(n,l),d!=null&&o.unshift(Jn(n,d,i))):a||(d=Gn(n,l),d!=null&&o.push(Jn(n,d,i)))),n=n.return}o.length!==0&&e.push({event:t,listeners:o})}var Wf=/\r\n?/g,Gf=/\u0000|\uFFFD/g;function Oi(e){return(typeof e=="string"?e:""+e).replace(Wf,`
`).replace(Gf,"")}function _s(e,t,n){if(t=Oi(t),Oi(e)!==t&&n)throw Error(U(425))}function ta(){}var Ll=null,Fl=null;function Dl(e,t){return e==="textarea"||e==="noscript"||typeof t.children=="string"||typeof t.children=="number"||typeof t.dangerouslySetInnerHTML=="object"&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var Ol=typeof setTimeout=="function"?setTimeout:void 0,Hf=typeof clearTimeout=="function"?clearTimeout:void 0,Ai=typeof Promise=="function"?Promise:void 0,Qf=typeof queueMicrotask=="function"?queueMicrotask:typeof Ai<"u"?function(e){return Ai.resolve(null).then(e).catch(Xf)}:Ol;function Xf(e){setTimeout(function(){throw e})}function el(e,t){var n=t,s=0;do{var a=n.nextSibling;if(e.removeChild(n),a&&a.nodeType===8)if(n=a.data,n==="/$"){if(s===0){e.removeChild(a),Xn(t);return}s--}else n!=="$"&&n!=="$?"&&n!=="$!"||s++;n=a}while(n);Xn(t)}function lr(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t==="$"||t==="$!"||t==="$?")break;if(t==="/$")return null}}return e}function $i(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="$"||n==="$!"||n==="$?"){if(t===0)return e;t--}else n==="/$"&&t++}e=e.previousSibling}return null}var fn=Math.random().toString(36).slice(2),zt="__reactFiber$"+fn,Zn="__reactProps$"+fn,Bt="__reactContainer$"+fn,Al="__reactEvents$"+fn,Yf="__reactListeners$"+fn,Kf="__reactHandles$"+fn;function kr(e){var t=e[zt];if(t)return t;for(var n=e.parentNode;n;){if(t=n[Bt]||n[zt]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=$i(e);e!==null;){if(n=e[zt])return n;e=$i(e)}return t}e=n,n=e.parentNode}return null}function us(e){return e=e[zt]||e[Bt],!e||e.tag!==5&&e.tag!==6&&e.tag!==13&&e.tag!==3?null:e}function Br(e){if(e.tag===5||e.tag===6)return e.stateNode;throw Error(U(33))}function ka(e){return e[Zn]||null}var $l=[],Wr=-1;function gr(e){return{current:e}}function be(e){0>Wr||(e.current=$l[Wr],$l[Wr]=null,Wr--)}function ye(e,t){Wr++,$l[Wr]=e.current,e.current=t}var pr={},Ve=gr(pr),Ke=gr(!1),Er=pr;function nn(e,t){var n=e.type.contextTypes;if(!n)return pr;var s=e.stateNode;if(s&&s.__reactInternalMemoizedUnmaskedChildContext===t)return s.__reactInternalMemoizedMaskedChildContext;var a={},l;for(l in n)a[l]=t[l];return s&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=t,e.__reactInternalMemoizedMaskedChildContext=a),a}function qe(e){return e=e.childContextTypes,e!=null}function ra(){be(Ke),be(Ve)}function Ui(e,t,n){if(Ve.current!==pr)throw Error(U(168));ye(Ve,t),ye(Ke,n)}function Od(e,t,n){var s=e.stateNode;if(t=t.childContextTypes,typeof s.getChildContext!="function")return n;s=s.getChildContext();for(var a in s)if(!(a in t))throw Error(U(108,Mp(e)||"Unknown",a));return Ne({},n,s)}function na(e){return e=(e=e.stateNode)&&e.__reactInternalMemoizedMergedChildContext||pr,Er=Ve.current,ye(Ve,e),ye(Ke,Ke.current),!0}function Vi(e,t,n){var s=e.stateNode;if(!s)throw Error(U(169));n?(e=Od(e,t,Er),s.__reactInternalMemoizedMergedChildContext=e,be(Ke),be(Ve),ye(Ve,e)):be(Ke),ye(Ke,n)}var Ft=null,Sa=!1,tl=!1;function Ad(e){Ft===null?Ft=[e]:Ft.push(e)}function qf(e){Sa=!0,Ad(e)}function vr(){if(!tl&&Ft!==null){tl=!0;var e=0,t=ge;try{var n=Ft;for(ge=1;e<n.length;e++){var s=n[e];do s=s(!0);while(s!==null)}Ft=null,Sa=!1}catch(a){throw Ft!==null&&(Ft=Ft.slice(e+1)),dd(jo,vr),a}finally{ge=t,tl=!1}}return null}var Gr=[],Hr=0,sa=null,aa=0,ct=[],dt=0,zr=null,Dt=1,Ot="";function br(e,t){Gr[Hr++]=aa,Gr[Hr++]=sa,sa=e,aa=t}function $d(e,t,n){ct[dt++]=Dt,ct[dt++]=Ot,ct[dt++]=zr,zr=e;var s=Dt;e=Ot;var a=32-bt(s)-1;s&=~(1<<a),n+=1;var l=32-bt(t)+a;if(30<l){var o=a-a%5;l=(s&(1<<o)-1).toString(32),s>>=o,a-=o,Dt=1<<32-bt(t)+a|n<<a|s,Ot=l+e}else Dt=1<<l|n<<a|s,Ot=e}function zo(e){e.return!==null&&(br(e,1),$d(e,1,0))}function To(e){for(;e===sa;)sa=Gr[--Hr],Gr[Hr]=null,aa=Gr[--Hr],Gr[Hr]=null;for(;e===zr;)zr=ct[--dt],ct[dt]=null,Ot=ct[--dt],ct[dt]=null,Dt=ct[--dt],ct[dt]=null}var st=null,nt=null,we=!1,jt=null;function Ud(e,t){var n=ut(5,null,null,0);n.elementType="DELETED",n.stateNode=t,n.return=e,t=e.deletions,t===null?(e.deletions=[n],e.flags|=16):t.push(n)}function Bi(e,t){switch(e.tag){case 5:var n=e.type;return t=t.nodeType!==1||n.toLowerCase()!==t.nodeName.toLowerCase()?null:t,t!==null?(e.stateNode=t,st=e,nt=lr(t.firstChild),!0):!1;case 6:return t=e.pendingProps===""||t.nodeType!==3?null:t,t!==null?(e.stateNode=t,st=e,nt=null,!0):!1;case 13:return t=t.nodeType!==8?null:t,t!==null?(n=zr!==null?{id:Dt,overflow:Ot}:null,e.memoizedState={dehydrated:t,treeContext:n,retryLane:1073741824},n=ut(18,null,null,0),n.stateNode=t,n.return=e,e.child=n,st=e,nt=null,!0):!1;default:return!1}}function Ul(e){return(e.mode&1)!==0&&(e.flags&128)===0}function Vl(e){if(we){var t=nt;if(t){var n=t;if(!Bi(e,t)){if(Ul(e))throw Error(U(418));t=lr(n.nextSibling);var s=st;t&&Bi(e,t)?Ud(s,n):(e.flags=e.flags&-4097|2,we=!1,st=e)}}else{if(Ul(e))throw Error(U(418));e.flags=e.flags&-4097|2,we=!1,st=e}}}function Wi(e){for(e=e.return;e!==null&&e.tag!==5&&e.tag!==3&&e.tag!==13;)e=e.return;st=e}function Es(e){if(e!==st)return!1;if(!we)return Wi(e),we=!0,!1;var t;if((t=e.tag!==3)&&!(t=e.tag!==5)&&(t=e.type,t=t!=="head"&&t!=="body"&&!Dl(e.type,e.memoizedProps)),t&&(t=nt)){if(Ul(e))throw Vd(),Error(U(418));for(;t;)Ud(e,t),t=lr(t.nextSibling)}if(Wi(e),e.tag===13){if(e=e.memoizedState,e=e!==null?e.dehydrated:null,!e)throw Error(U(317));e:{for(e=e.nextSibling,t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="/$"){if(t===0){nt=lr(e.nextSibling);break e}t--}else n!=="$"&&n!=="$!"&&n!=="$?"||t++}e=e.nextSibling}nt=null}}else nt=st?lr(e.stateNode.nextSibling):null;return!0}function Vd(){for(var e=nt;e;)e=lr(e.nextSibling)}function sn(){nt=st=null,we=!1}function Po(e){jt===null?jt=[e]:jt.push(e)}var Jf=Qt.ReactCurrentBatchConfig;function Nn(e,t,n){if(e=n.ref,e!==null&&typeof e!="function"&&typeof e!="object"){if(n._owner){if(n=n._owner,n){if(n.tag!==1)throw Error(U(309));var s=n.stateNode}if(!s)throw Error(U(147,e));var a=s,l=""+e;return t!==null&&t.ref!==null&&typeof t.ref=="function"&&t.ref._stringRef===l?t.ref:(t=function(o){var i=a.refs;o===null?delete i[l]:i[l]=o},t._stringRef=l,t)}if(typeof e!="string")throw Error(U(284));if(!n._owner)throw Error(U(290,e))}return e}function zs(e,t){throw e=Object.prototype.toString.call(t),Error(U(31,e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e))}function Gi(e){var t=e._init;return t(e._payload)}function Bd(e){function t(f,p){if(e){var m=f.deletions;m===null?(f.deletions=[p],f.flags|=16):m.push(p)}}function n(f,p){if(!e)return null;for(;p!==null;)t(f,p),p=p.sibling;return null}function s(f,p){for(f=new Map;p!==null;)p.key!==null?f.set(p.key,p):f.set(p.index,p),p=p.sibling;return f}function a(f,p){return f=dr(f,p),f.index=0,f.sibling=null,f}function l(f,p,m){return f.index=m,e?(m=f.alternate,m!==null?(m=m.index,m<p?(f.flags|=2,p):m):(f.flags|=2,p)):(f.flags|=1048576,p)}function o(f){return e&&f.alternate===null&&(f.flags|=2),f}function i(f,p,m,h){return p===null||p.tag!==6?(p=il(m,f.mode,h),p.return=f,p):(p=a(p,m),p.return=f,p)}function d(f,p,m,h){var j=m.type;return j===Ar?y(f,p,m.props.children,h,m.key):p!==null&&(p.elementType===j||typeof j=="object"&&j!==null&&j.$$typeof===qt&&Gi(j)===p.type)?(h=a(p,m.props),h.ref=Nn(f,p,m),h.return=f,h):(h=Hs(m.type,m.key,m.props,null,f.mode,h),h.ref=Nn(f,p,m),h.return=f,h)}function u(f,p,m,h){return p===null||p.tag!==4||p.stateNode.containerInfo!==m.containerInfo||p.stateNode.implementation!==m.implementation?(p=cl(m,f.mode,h),p.return=f,p):(p=a(p,m.children||[]),p.return=f,p)}function y(f,p,m,h,j){return p===null||p.tag!==7?(p=_r(m,f.mode,h,j),p.return=f,p):(p=a(p,m),p.return=f,p)}function g(f,p,m){if(typeof p=="string"&&p!==""||typeof p=="number")return p=il(""+p,f.mode,m),p.return=f,p;if(typeof p=="object"&&p!==null){switch(p.$$typeof){case vs:return m=Hs(p.type,p.key,p.props,null,f.mode,m),m.ref=Nn(f,null,p),m.return=f,m;case Or:return p=cl(p,f.mode,m),p.return=f,p;case qt:var h=p._init;return g(f,h(p._payload),m)}if(Pn(p)||jn(p))return p=_r(p,f.mode,m,null),p.return=f,p;zs(f,p)}return null}function x(f,p,m,h){var j=p!==null?p.key:null;if(typeof m=="string"&&m!==""||typeof m=="number")return j!==null?null:i(f,p,""+m,h);if(typeof m=="object"&&m!==null){switch(m.$$typeof){case vs:return m.key===j?d(f,p,m,h):null;case Or:return m.key===j?u(f,p,m,h):null;case qt:return j=m._init,x(f,p,j(m._payload),h)}if(Pn(m)||jn(m))return j!==null?null:y(f,p,m,h,null);zs(f,m)}return null}function k(f,p,m,h,j){if(typeof h=="string"&&h!==""||typeof h=="number")return f=f.get(m)||null,i(p,f,""+h,j);if(typeof h=="object"&&h!==null){switch(h.$$typeof){case vs:return f=f.get(h.key===null?m:h.key)||null,d(p,f,h,j);case Or:return f=f.get(h.key===null?m:h.key)||null,u(p,f,h,j);case qt:var _=h._init;return k(f,p,m,_(h._payload),j)}if(Pn(h)||jn(h))return f=f.get(m)||null,y(p,f,h,j,null);zs(p,h)}return null}function S(f,p,m,h){for(var j=null,_=null,P=p,I=p=0,G=null;P!==null&&I<m.length;I++){P.index>I?(G=P,P=null):G=P.sibling;var H=x(f,P,m[I],h);if(H===null){P===null&&(P=G);break}e&&P&&H.alternate===null&&t(f,P),p=l(H,p,I),_===null?j=H:_.sibling=H,_=H,P=G}if(I===m.length)return n(f,P),we&&br(f,I),j;if(P===null){for(;I<m.length;I++)P=g(f,m[I],h),P!==null&&(p=l(P,p,I),_===null?j=P:_.sibling=P,_=P);return we&&br(f,I),j}for(P=s(f,P);I<m.length;I++)G=k(P,f,I,m[I],h),G!==null&&(e&&G.alternate!==null&&P.delete(G.key===null?I:G.key),p=l(G,p,I),_===null?j=G:_.sibling=G,_=G);return e&&P.forEach(function(N){return t(f,N)}),we&&br(f,I),j}function z(f,p,m,h){var j=jn(m);if(typeof j!="function")throw Error(U(150));if(m=j.call(m),m==null)throw Error(U(151));for(var _=j=null,P=p,I=p=0,G=null,H=m.next();P!==null&&!H.done;I++,H=m.next()){P.index>I?(G=P,P=null):G=P.sibling;var N=x(f,P,H.value,h);if(N===null){P===null&&(P=G);break}e&&P&&N.alternate===null&&t(f,P),p=l(N,p,I),_===null?j=N:_.sibling=N,_=N,P=G}if(H.done)return n(f,P),we&&br(f,I),j;if(P===null){for(;!H.done;I++,H=m.next())H=g(f,H.value,h),H!==null&&(p=l(H,p,I),_===null?j=H:_.sibling=H,_=H);return we&&br(f,I),j}for(P=s(f,P);!H.done;I++,H=m.next())H=k(P,f,I,H.value,h),H!==null&&(e&&H.alternate!==null&&P.delete(H.key===null?I:H.key),p=l(H,p,I),_===null?j=H:_.sibling=H,_=H);return e&&P.forEach(function(C){return t(f,C)}),we&&br(f,I),j}function R(f,p,m,h){if(typeof m=="object"&&m!==null&&m.type===Ar&&m.key===null&&(m=m.props.children),typeof m=="object"&&m!==null){switch(m.$$typeof){case vs:e:{for(var j=m.key,_=p;_!==null;){if(_.key===j){if(j=m.type,j===Ar){if(_.tag===7){n(f,_.sibling),p=a(_,m.props.children),p.return=f,f=p;break e}}else if(_.elementType===j||typeof j=="object"&&j!==null&&j.$$typeof===qt&&Gi(j)===_.type){n(f,_.sibling),p=a(_,m.props),p.ref=Nn(f,_,m),p.return=f,f=p;break e}n(f,_);break}else t(f,_);_=_.sibling}m.type===Ar?(p=_r(m.props.children,f.mode,h,m.key),p.return=f,f=p):(h=Hs(m.type,m.key,m.props,null,f.mode,h),h.ref=Nn(f,p,m),h.return=f,f=h)}return o(f);case Or:e:{for(_=m.key;p!==null;){if(p.key===_)if(p.tag===4&&p.stateNode.containerInfo===m.containerInfo&&p.stateNode.implementation===m.implementation){n(f,p.sibling),p=a(p,m.children||[]),p.return=f,f=p;break e}else{n(f,p);break}else t(f,p);p=p.sibling}p=cl(m,f.mode,h),p.return=f,f=p}return o(f);case qt:return _=m._init,R(f,p,_(m._payload),h)}if(Pn(m))return S(f,p,m,h);if(jn(m))return z(f,p,m,h);zs(f,m)}return typeof m=="string"&&m!==""||typeof m=="number"?(m=""+m,p!==null&&p.tag===6?(n(f,p.sibling),p=a(p,m),p.return=f,f=p):(n(f,p),p=il(m,f.mode,h),p.return=f,f=p),o(f)):n(f,p)}return R}var an=Bd(!0),Wd=Bd(!1),la=gr(null),oa=null,Qr=null,Io=null;function Mo(){Io=Qr=oa=null}function Ro(e){var t=la.current;be(la),e._currentValue=t}function Bl(e,t,n){for(;e!==null;){var s=e.alternate;if((e.childLanes&t)!==t?(e.childLanes|=t,s!==null&&(s.childLanes|=t)):s!==null&&(s.childLanes&t)!==t&&(s.childLanes|=t),e===n)break;e=e.return}}function en(e,t){oa=e,Io=Qr=null,e=e.dependencies,e!==null&&e.firstContext!==null&&(e.lanes&t&&(Ye=!0),e.firstContext=null)}function ft(e){var t=e._currentValue;if(Io!==e)if(e={context:e,memoizedValue:t,next:null},Qr===null){if(oa===null)throw Error(U(308));Qr=e,oa.dependencies={lanes:0,firstContext:e}}else Qr=Qr.next=e;return t}var Sr=null;function Lo(e){Sr===null?Sr=[e]:Sr.push(e)}function Gd(e,t,n,s){var a=t.interleaved;return a===null?(n.next=n,Lo(t)):(n.next=a.next,a.next=n),t.interleaved=n,Wt(e,s)}function Wt(e,t){e.lanes|=t;var n=e.alternate;for(n!==null&&(n.lanes|=t),n=e,e=e.return;e!==null;)e.childLanes|=t,n=e.alternate,n!==null&&(n.childLanes|=t),n=e,e=e.return;return n.tag===3?n.stateNode:null}var Jt=!1;function Fo(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,interleaved:null,lanes:0},effects:null}}function Hd(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,effects:e.effects})}function $t(e,t){return{eventTime:e,lane:t,tag:0,payload:null,callback:null,next:null}}function or(e,t,n){var s=e.updateQueue;if(s===null)return null;if(s=s.shared,fe&2){var a=s.pending;return a===null?t.next=t:(t.next=a.next,a.next=t),s.pending=t,Wt(e,n)}return a=s.interleaved,a===null?(t.next=t,Lo(s)):(t.next=a.next,a.next=t),s.interleaved=t,Wt(e,n)}function $s(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,(n&4194240)!==0)){var s=t.lanes;s&=e.pendingLanes,n|=s,t.lanes=n,bo(e,n)}}function Hi(e,t){var n=e.updateQueue,s=e.alternate;if(s!==null&&(s=s.updateQueue,n===s)){var a=null,l=null;if(n=n.firstBaseUpdate,n!==null){do{var o={eventTime:n.eventTime,lane:n.lane,tag:n.tag,payload:n.payload,callback:n.callback,next:null};l===null?a=l=o:l=l.next=o,n=n.next}while(n!==null);l===null?a=l=t:l=l.next=t}else a=l=t;n={baseState:s.baseState,firstBaseUpdate:a,lastBaseUpdate:l,shared:s.shared,effects:s.effects},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}function ia(e,t,n,s){var a=e.updateQueue;Jt=!1;var l=a.firstBaseUpdate,o=a.lastBaseUpdate,i=a.shared.pending;if(i!==null){a.shared.pending=null;var d=i,u=d.next;d.next=null,o===null?l=u:o.next=u,o=d;var y=e.alternate;y!==null&&(y=y.updateQueue,i=y.lastBaseUpdate,i!==o&&(i===null?y.firstBaseUpdate=u:i.next=u,y.lastBaseUpdate=d))}if(l!==null){var g=a.baseState;o=0,y=u=d=null,i=l;do{var x=i.lane,k=i.eventTime;if((s&x)===x){y!==null&&(y=y.next={eventTime:k,lane:0,tag:i.tag,payload:i.payload,callback:i.callback,next:null});e:{var S=e,z=i;switch(x=t,k=n,z.tag){case 1:if(S=z.payload,typeof S=="function"){g=S.call(k,g,x);break e}g=S;break e;case 3:S.flags=S.flags&-65537|128;case 0:if(S=z.payload,x=typeof S=="function"?S.call(k,g,x):S,x==null)break e;g=Ne({},g,x);break e;case 2:Jt=!0}}i.callback!==null&&i.lane!==0&&(e.flags|=64,x=a.effects,x===null?a.effects=[i]:x.push(i))}else k={eventTime:k,lane:x,tag:i.tag,payload:i.payload,callback:i.callback,next:null},y===null?(u=y=k,d=g):y=y.next=k,o|=x;if(i=i.next,i===null){if(i=a.shared.pending,i===null)break;x=i,i=x.next,x.next=null,a.lastBaseUpdate=x,a.shared.pending=null}}while(!0);if(y===null&&(d=g),a.baseState=d,a.firstBaseUpdate=u,a.lastBaseUpdate=y,t=a.shared.interleaved,t!==null){a=t;do o|=a.lane,a=a.next;while(a!==t)}else l===null&&(a.shared.lanes=0);Pr|=o,e.lanes=o,e.memoizedState=g}}function Qi(e,t,n){if(e=t.effects,t.effects=null,e!==null)for(t=0;t<e.length;t++){var s=e[t],a=s.callback;if(a!==null){if(s.callback=null,s=n,typeof a!="function")throw Error(U(191,a));a.call(s)}}}var ps={},Pt=gr(ps),es=gr(ps),ts=gr(ps);function Nr(e){if(e===ps)throw Error(U(174));return e}function Do(e,t){switch(ye(ts,t),ye(es,e),ye(Pt,ps),e=t.nodeType,e){case 9:case 11:t=(t=t.documentElement)?t.namespaceURI:kl(null,"");break;default:e=e===8?t.parentNode:t,t=e.namespaceURI||null,e=e.tagName,t=kl(t,e)}be(Pt),ye(Pt,t)}function ln(){be(Pt),be(es),be(ts)}function Qd(e){Nr(ts.current);var t=Nr(Pt.current),n=kl(t,e.type);t!==n&&(ye(es,e),ye(Pt,n))}function Oo(e){es.current===e&&(be(Pt),be(es))}var ke=gr(0);function ca(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||n.data==="$?"||n.data==="$!"))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==void 0){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var rl=[];function Ao(){for(var e=0;e<rl.length;e++)rl[e]._workInProgressVersionPrimary=null;rl.length=0}var Us=Qt.ReactCurrentDispatcher,nl=Qt.ReactCurrentBatchConfig,Tr=0,Se=null,Te=null,Ie=null,da=!1,An=!1,rs=0,Zf=0;function Ae(){throw Error(U(321))}function $o(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!kt(e[n],t[n]))return!1;return!0}function Uo(e,t,n,s,a,l){if(Tr=l,Se=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,Us.current=e===null||e.memoizedState===null?nm:sm,e=n(s,a),An){l=0;do{if(An=!1,rs=0,25<=l)throw Error(U(301));l+=1,Ie=Te=null,t.updateQueue=null,Us.current=am,e=n(s,a)}while(An)}if(Us.current=ua,t=Te!==null&&Te.next!==null,Tr=0,Ie=Te=Se=null,da=!1,t)throw Error(U(300));return e}function Vo(){var e=rs!==0;return rs=0,e}function Et(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return Ie===null?Se.memoizedState=Ie=e:Ie=Ie.next=e,Ie}function mt(){if(Te===null){var e=Se.alternate;e=e!==null?e.memoizedState:null}else e=Te.next;var t=Ie===null?Se.memoizedState:Ie.next;if(t!==null)Ie=t,Te=e;else{if(e===null)throw Error(U(310));Te=e,e={memoizedState:Te.memoizedState,baseState:Te.baseState,baseQueue:Te.baseQueue,queue:Te.queue,next:null},Ie===null?Se.memoizedState=Ie=e:Ie=Ie.next=e}return Ie}function ns(e,t){return typeof t=="function"?t(e):t}function sl(e){var t=mt(),n=t.queue;if(n===null)throw Error(U(311));n.lastRenderedReducer=e;var s=Te,a=s.baseQueue,l=n.pending;if(l!==null){if(a!==null){var o=a.next;a.next=l.next,l.next=o}s.baseQueue=a=l,n.pending=null}if(a!==null){l=a.next,s=s.baseState;var i=o=null,d=null,u=l;do{var y=u.lane;if((Tr&y)===y)d!==null&&(d=d.next={lane:0,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),s=u.hasEagerState?u.eagerState:e(s,u.action);else{var g={lane:y,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null};d===null?(i=d=g,o=s):d=d.next=g,Se.lanes|=y,Pr|=y}u=u.next}while(u!==null&&u!==l);d===null?o=s:d.next=i,kt(s,t.memoizedState)||(Ye=!0),t.memoizedState=s,t.baseState=o,t.baseQueue=d,n.lastRenderedState=s}if(e=n.interleaved,e!==null){a=e;do l=a.lane,Se.lanes|=l,Pr|=l,a=a.next;while(a!==e)}else a===null&&(n.lanes=0);return[t.memoizedState,n.dispatch]}function al(e){var t=mt(),n=t.queue;if(n===null)throw Error(U(311));n.lastRenderedReducer=e;var s=n.dispatch,a=n.pending,l=t.memoizedState;if(a!==null){n.pending=null;var o=a=a.next;do l=e(l,o.action),o=o.next;while(o!==a);kt(l,t.memoizedState)||(Ye=!0),t.memoizedState=l,t.baseQueue===null&&(t.baseState=l),n.lastRenderedState=l}return[l,s]}function Xd(){}function Yd(e,t){var n=Se,s=mt(),a=t(),l=!kt(s.memoizedState,a);if(l&&(s.memoizedState=a,Ye=!0),s=s.queue,Bo(Jd.bind(null,n,s,e),[e]),s.getSnapshot!==t||l||Ie!==null&&Ie.memoizedState.tag&1){if(n.flags|=2048,ss(9,qd.bind(null,n,s,a,t),void 0,null),Me===null)throw Error(U(349));Tr&30||Kd(n,t,a)}return a}function Kd(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=Se.updateQueue,t===null?(t={lastEffect:null,stores:null},Se.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function qd(e,t,n,s){t.value=n,t.getSnapshot=s,Zd(t)&&eu(e)}function Jd(e,t,n){return n(function(){Zd(t)&&eu(e)})}function Zd(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!kt(e,n)}catch{return!0}}function eu(e){var t=Wt(e,1);t!==null&&wt(t,e,1,-1)}function Xi(e){var t=Et();return typeof e=="function"&&(e=e()),t.memoizedState=t.baseState=e,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:ns,lastRenderedState:e},t.queue=e,e=e.dispatch=rm.bind(null,Se,e),[t.memoizedState,e]}function ss(e,t,n,s){return e={tag:e,create:t,destroy:n,deps:s,next:null},t=Se.updateQueue,t===null?(t={lastEffect:null,stores:null},Se.updateQueue=t,t.lastEffect=e.next=e):(n=t.lastEffect,n===null?t.lastEffect=e.next=e:(s=n.next,n.next=e,e.next=s,t.lastEffect=e)),e}function tu(){return mt().memoizedState}function Vs(e,t,n,s){var a=Et();Se.flags|=e,a.memoizedState=ss(1|t,n,void 0,s===void 0?null:s)}function Na(e,t,n,s){var a=mt();s=s===void 0?null:s;var l=void 0;if(Te!==null){var o=Te.memoizedState;if(l=o.destroy,s!==null&&$o(s,o.deps)){a.memoizedState=ss(t,n,l,s);return}}Se.flags|=e,a.memoizedState=ss(1|t,n,l,s)}function Yi(e,t){return Vs(8390656,8,e,t)}function Bo(e,t){return Na(2048,8,e,t)}function ru(e,t){return Na(4,2,e,t)}function nu(e,t){return Na(4,4,e,t)}function su(e,t){if(typeof t=="function")return e=e(),t(e),function(){t(null)};if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function au(e,t,n){return n=n!=null?n.concat([e]):null,Na(4,4,su.bind(null,t,e),n)}function Wo(){}function lu(e,t){var n=mt();t=t===void 0?null:t;var s=n.memoizedState;return s!==null&&t!==null&&$o(t,s[1])?s[0]:(n.memoizedState=[e,t],e)}function ou(e,t){var n=mt();t=t===void 0?null:t;var s=n.memoizedState;return s!==null&&t!==null&&$o(t,s[1])?s[0]:(e=e(),n.memoizedState=[e,t],e)}function iu(e,t,n){return Tr&21?(kt(n,t)||(n=fd(),Se.lanes|=n,Pr|=n,e.baseState=!0),t):(e.baseState&&(e.baseState=!1,Ye=!0),e.memoizedState=n)}function em(e,t){var n=ge;ge=n!==0&&4>n?n:4,e(!0);var s=nl.transition;nl.transition={};try{e(!1),t()}finally{ge=n,nl.transition=s}}function cu(){return mt().memoizedState}function tm(e,t,n){var s=cr(e);if(n={lane:s,action:n,hasEagerState:!1,eagerState:null,next:null},du(e))uu(t,n);else if(n=Gd(e,t,n,s),n!==null){var a=Ge();wt(n,e,s,a),pu(n,t,s)}}function rm(e,t,n){var s=cr(e),a={lane:s,action:n,hasEagerState:!1,eagerState:null,next:null};if(du(e))uu(t,a);else{var l=e.alternate;if(e.lanes===0&&(l===null||l.lanes===0)&&(l=t.lastRenderedReducer,l!==null))try{var o=t.lastRenderedState,i=l(o,n);if(a.hasEagerState=!0,a.eagerState=i,kt(i,o)){var d=t.interleaved;d===null?(a.next=a,Lo(t)):(a.next=d.next,d.next=a),t.interleaved=a;return}}catch{}finally{}n=Gd(e,t,a,s),n!==null&&(a=Ge(),wt(n,e,s,a),pu(n,t,s))}}function du(e){var t=e.alternate;return e===Se||t!==null&&t===Se}function uu(e,t){An=da=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function pu(e,t,n){if(n&4194240){var s=t.lanes;s&=e.pendingLanes,n|=s,t.lanes=n,bo(e,n)}}var ua={readContext:ft,useCallback:Ae,useContext:Ae,useEffect:Ae,useImperativeHandle:Ae,useInsertionEffect:Ae,useLayoutEffect:Ae,useMemo:Ae,useReducer:Ae,useRef:Ae,useState:Ae,useDebugValue:Ae,useDeferredValue:Ae,useTransition:Ae,useMutableSource:Ae,useSyncExternalStore:Ae,useId:Ae,unstable_isNewReconciler:!1},nm={readContext:ft,useCallback:function(e,t){return Et().memoizedState=[e,t===void 0?null:t],e},useContext:ft,useEffect:Yi,useImperativeHandle:function(e,t,n){return n=n!=null?n.concat([e]):null,Vs(4194308,4,su.bind(null,t,e),n)},useLayoutEffect:function(e,t){return Vs(4194308,4,e,t)},useInsertionEffect:function(e,t){return Vs(4,2,e,t)},useMemo:function(e,t){var n=Et();return t=t===void 0?null:t,e=e(),n.memoizedState=[e,t],e},useReducer:function(e,t,n){var s=Et();return t=n!==void 0?n(t):t,s.memoizedState=s.baseState=t,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:t},s.queue=e,e=e.dispatch=tm.bind(null,Se,e),[s.memoizedState,e]},useRef:function(e){var t=Et();return e={current:e},t.memoizedState=e},useState:Xi,useDebugValue:Wo,useDeferredValue:function(e){return Et().memoizedState=e},useTransition:function(){var e=Xi(!1),t=e[0];return e=em.bind(null,e[1]),Et().memoizedState=e,[t,e]},useMutableSource:function(){},useSyncExternalStore:function(e,t,n){var s=Se,a=Et();if(we){if(n===void 0)throw Error(U(407));n=n()}else{if(n=t(),Me===null)throw Error(U(349));Tr&30||Kd(s,t,n)}a.memoizedState=n;var l={value:n,getSnapshot:t};return a.queue=l,Yi(Jd.bind(null,s,l,e),[e]),s.flags|=2048,ss(9,qd.bind(null,s,l,n,t),void 0,null),n},useId:function(){var e=Et(),t=Me.identifierPrefix;if(we){var n=Ot,s=Dt;n=(s&~(1<<32-bt(s)-1)).toString(32)+n,t=":"+t+"R"+n,n=rs++,0<n&&(t+="H"+n.toString(32)),t+=":"}else n=Zf++,t=":"+t+"r"+n.toString(32)+":";return e.memoizedState=t},unstable_isNewReconciler:!1},sm={readContext:ft,useCallback:lu,useContext:ft,useEffect:Bo,useImperativeHandle:au,useInsertionEffect:ru,useLayoutEffect:nu,useMemo:ou,useReducer:sl,useRef:tu,useState:function(){return sl(ns)},useDebugValue:Wo,useDeferredValue:function(e){var t=mt();return iu(t,Te.memoizedState,e)},useTransition:function(){var e=sl(ns)[0],t=mt().memoizedState;return[e,t]},useMutableSource:Xd,useSyncExternalStore:Yd,useId:cu,unstable_isNewReconciler:!1},am={readContext:ft,useCallback:lu,useContext:ft,useEffect:Bo,useImperativeHandle:au,useInsertionEffect:ru,useLayoutEffect:nu,useMemo:ou,useReducer:al,useRef:tu,useState:function(){return al(ns)},useDebugValue:Wo,useDeferredValue:function(e){var t=mt();return Te===null?t.memoizedState=e:iu(t,Te.memoizedState,e)},useTransition:function(){var e=al(ns)[0],t=mt().memoizedState;return[e,t]},useMutableSource:Xd,useSyncExternalStore:Yd,useId:cu,unstable_isNewReconciler:!1};function vt(e,t){if(e&&e.defaultProps){t=Ne({},t),e=e.defaultProps;for(var n in e)t[n]===void 0&&(t[n]=e[n]);return t}return t}function Wl(e,t,n,s){t=e.memoizedState,n=n(s,t),n=n==null?t:Ne({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var Ca={isMounted:function(e){return(e=e._reactInternals)?Rr(e)===e:!1},enqueueSetState:function(e,t,n){e=e._reactInternals;var s=Ge(),a=cr(e),l=$t(s,a);l.payload=t,n!=null&&(l.callback=n),t=or(e,l,a),t!==null&&(wt(t,e,a,s),$s(t,e,a))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var s=Ge(),a=cr(e),l=$t(s,a);l.tag=1,l.payload=t,n!=null&&(l.callback=n),t=or(e,l,a),t!==null&&(wt(t,e,a,s),$s(t,e,a))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=Ge(),s=cr(e),a=$t(n,s);a.tag=2,t!=null&&(a.callback=t),t=or(e,a,s),t!==null&&(wt(t,e,s,n),$s(t,e,s))}};function Ki(e,t,n,s,a,l,o){return e=e.stateNode,typeof e.shouldComponentUpdate=="function"?e.shouldComponentUpdate(s,l,o):t.prototype&&t.prototype.isPureReactComponent?!Kn(n,s)||!Kn(a,l):!0}function fu(e,t,n){var s=!1,a=pr,l=t.contextType;return typeof l=="object"&&l!==null?l=ft(l):(a=qe(t)?Er:Ve.current,s=t.contextTypes,l=(s=s!=null)?nn(e,a):pr),t=new t(n,l),e.memoizedState=t.state!==null&&t.state!==void 0?t.state:null,t.updater=Ca,e.stateNode=t,t._reactInternals=e,s&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=a,e.__reactInternalMemoizedMaskedChildContext=l),t}function qi(e,t,n,s){e=t.state,typeof t.componentWillReceiveProps=="function"&&t.componentWillReceiveProps(n,s),typeof t.UNSAFE_componentWillReceiveProps=="function"&&t.UNSAFE_componentWillReceiveProps(n,s),t.state!==e&&Ca.enqueueReplaceState(t,t.state,null)}function Gl(e,t,n,s){var a=e.stateNode;a.props=n,a.state=e.memoizedState,a.refs={},Fo(e);var l=t.contextType;typeof l=="object"&&l!==null?a.context=ft(l):(l=qe(t)?Er:Ve.current,a.context=nn(e,l)),a.state=e.memoizedState,l=t.getDerivedStateFromProps,typeof l=="function"&&(Wl(e,t,l,n),a.state=e.memoizedState),typeof t.getDerivedStateFromProps=="function"||typeof a.getSnapshotBeforeUpdate=="function"||typeof a.UNSAFE_componentWillMount!="function"&&typeof a.componentWillMount!="function"||(t=a.state,typeof a.componentWillMount=="function"&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount=="function"&&a.UNSAFE_componentWillMount(),t!==a.state&&Ca.enqueueReplaceState(a,a.state,null),ia(e,n,a,s),a.state=e.memoizedState),typeof a.componentDidMount=="function"&&(e.flags|=4194308)}function on(e,t){try{var n="",s=t;do n+=Ip(s),s=s.return;while(s);var a=n}catch(l){a=`
Error generating stack: `+l.message+`
`+l.stack}return{value:e,source:t,stack:a,digest:null}}function ll(e,t,n){return{value:e,source:null,stack:n??null,digest:t??null}}function Hl(e,t){try{console.error(t.value)}catch(n){setTimeout(function(){throw n})}}var lm=typeof WeakMap=="function"?WeakMap:Map;function mu(e,t,n){n=$t(-1,n),n.tag=3,n.payload={element:null};var s=t.value;return n.callback=function(){fa||(fa=!0,ro=s),Hl(e,t)},n}function hu(e,t,n){n=$t(-1,n),n.tag=3;var s=e.type.getDerivedStateFromError;if(typeof s=="function"){var a=t.value;n.payload=function(){return s(a)},n.callback=function(){Hl(e,t)}}var l=e.stateNode;return l!==null&&typeof l.componentDidCatch=="function"&&(n.callback=function(){Hl(e,t),typeof s!="function"&&(ir===null?ir=new Set([this]):ir.add(this));var o=t.stack;this.componentDidCatch(t.value,{componentStack:o!==null?o:""})}),n}function Ji(e,t,n){var s=e.pingCache;if(s===null){s=e.pingCache=new lm;var a=new Set;s.set(t,a)}else a=s.get(t),a===void 0&&(a=new Set,s.set(t,a));a.has(n)||(a.add(n),e=jm.bind(null,e,t,n),t.then(e,e))}function Zi(e){do{var t;if((t=e.tag===13)&&(t=e.memoizedState,t=t!==null?t.dehydrated!==null:!0),t)return e;e=e.return}while(e!==null);return null}function ec(e,t,n,s,a){return e.mode&1?(e.flags|=65536,e.lanes=a,e):(e===t?e.flags|=65536:(e.flags|=128,n.flags|=131072,n.flags&=-52805,n.tag===1&&(n.alternate===null?n.tag=17:(t=$t(-1,1),t.tag=2,or(n,t,1))),n.lanes|=1),e)}var om=Qt.ReactCurrentOwner,Ye=!1;function We(e,t,n,s){t.child=e===null?Wd(t,null,n,s):an(t,e.child,n,s)}function tc(e,t,n,s,a){n=n.render;var l=t.ref;return en(t,a),s=Uo(e,t,n,s,l,a),n=Vo(),e!==null&&!Ye?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~a,Gt(e,t,a)):(we&&n&&zo(t),t.flags|=1,We(e,t,s,a),t.child)}function rc(e,t,n,s,a){if(e===null){var l=n.type;return typeof l=="function"&&!Jo(l)&&l.defaultProps===void 0&&n.compare===null&&n.defaultProps===void 0?(t.tag=15,t.type=l,xu(e,t,l,s,a)):(e=Hs(n.type,null,s,t,t.mode,a),e.ref=t.ref,e.return=t,t.child=e)}if(l=e.child,!(e.lanes&a)){var o=l.memoizedProps;if(n=n.compare,n=n!==null?n:Kn,n(o,s)&&e.ref===t.ref)return Gt(e,t,a)}return t.flags|=1,e=dr(l,s),e.ref=t.ref,e.return=t,t.child=e}function xu(e,t,n,s,a){if(e!==null){var l=e.memoizedProps;if(Kn(l,s)&&e.ref===t.ref)if(Ye=!1,t.pendingProps=s=l,(e.lanes&a)!==0)e.flags&131072&&(Ye=!0);else return t.lanes=e.lanes,Gt(e,t,a)}return Ql(e,t,n,s,a)}function gu(e,t,n){var s=t.pendingProps,a=s.children,l=e!==null?e.memoizedState:null;if(s.mode==="hidden")if(!(t.mode&1))t.memoizedState={baseLanes:0,cachePool:null,transitions:null},ye(Yr,rt),rt|=n;else{if(!(n&1073741824))return e=l!==null?l.baseLanes|n:n,t.lanes=t.childLanes=1073741824,t.memoizedState={baseLanes:e,cachePool:null,transitions:null},t.updateQueue=null,ye(Yr,rt),rt|=e,null;t.memoizedState={baseLanes:0,cachePool:null,transitions:null},s=l!==null?l.baseLanes:n,ye(Yr,rt),rt|=s}else l!==null?(s=l.baseLanes|n,t.memoizedState=null):s=n,ye(Yr,rt),rt|=s;return We(e,t,a,n),t.child}function vu(e,t){var n=t.ref;(e===null&&n!==null||e!==null&&e.ref!==n)&&(t.flags|=512,t.flags|=2097152)}function Ql(e,t,n,s,a){var l=qe(n)?Er:Ve.current;return l=nn(t,l),en(t,a),n=Uo(e,t,n,s,l,a),s=Vo(),e!==null&&!Ye?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~a,Gt(e,t,a)):(we&&s&&zo(t),t.flags|=1,We(e,t,n,a),t.child)}function nc(e,t,n,s,a){if(qe(n)){var l=!0;na(t)}else l=!1;if(en(t,a),t.stateNode===null)Bs(e,t),fu(t,n,s),Gl(t,n,s,a),s=!0;else if(e===null){var o=t.stateNode,i=t.memoizedProps;o.props=i;var d=o.context,u=n.contextType;typeof u=="object"&&u!==null?u=ft(u):(u=qe(n)?Er:Ve.current,u=nn(t,u));var y=n.getDerivedStateFromProps,g=typeof y=="function"||typeof o.getSnapshotBeforeUpdate=="function";g||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(i!==s||d!==u)&&qi(t,o,s,u),Jt=!1;var x=t.memoizedState;o.state=x,ia(t,s,o,a),d=t.memoizedState,i!==s||x!==d||Ke.current||Jt?(typeof y=="function"&&(Wl(t,n,y,s),d=t.memoizedState),(i=Jt||Ki(t,n,i,s,x,d,u))?(g||typeof o.UNSAFE_componentWillMount!="function"&&typeof o.componentWillMount!="function"||(typeof o.componentWillMount=="function"&&o.componentWillMount(),typeof o.UNSAFE_componentWillMount=="function"&&o.UNSAFE_componentWillMount()),typeof o.componentDidMount=="function"&&(t.flags|=4194308)):(typeof o.componentDidMount=="function"&&(t.flags|=4194308),t.memoizedProps=s,t.memoizedState=d),o.props=s,o.state=d,o.context=u,s=i):(typeof o.componentDidMount=="function"&&(t.flags|=4194308),s=!1)}else{o=t.stateNode,Hd(e,t),i=t.memoizedProps,u=t.type===t.elementType?i:vt(t.type,i),o.props=u,g=t.pendingProps,x=o.context,d=n.contextType,typeof d=="object"&&d!==null?d=ft(d):(d=qe(n)?Er:Ve.current,d=nn(t,d));var k=n.getDerivedStateFromProps;(y=typeof k=="function"||typeof o.getSnapshotBeforeUpdate=="function")||typeof o.UNSAFE_componentWillReceiveProps!="function"&&typeof o.componentWillReceiveProps!="function"||(i!==g||x!==d)&&qi(t,o,s,d),Jt=!1,x=t.memoizedState,o.state=x,ia(t,s,o,a);var S=t.memoizedState;i!==g||x!==S||Ke.current||Jt?(typeof k=="function"&&(Wl(t,n,k,s),S=t.memoizedState),(u=Jt||Ki(t,n,u,s,x,S,d)||!1)?(y||typeof o.UNSAFE_componentWillUpdate!="function"&&typeof o.componentWillUpdate!="function"||(typeof o.componentWillUpdate=="function"&&o.componentWillUpdate(s,S,d),typeof o.UNSAFE_componentWillUpdate=="function"&&o.UNSAFE_componentWillUpdate(s,S,d)),typeof o.componentDidUpdate=="function"&&(t.flags|=4),typeof o.getSnapshotBeforeUpdate=="function"&&(t.flags|=1024)):(typeof o.componentDidUpdate!="function"||i===e.memoizedProps&&x===e.memoizedState||(t.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||i===e.memoizedProps&&x===e.memoizedState||(t.flags|=1024),t.memoizedProps=s,t.memoizedState=S),o.props=s,o.state=S,o.context=d,s=u):(typeof o.componentDidUpdate!="function"||i===e.memoizedProps&&x===e.memoizedState||(t.flags|=4),typeof o.getSnapshotBeforeUpdate!="function"||i===e.memoizedProps&&x===e.memoizedState||(t.flags|=1024),s=!1)}return Xl(e,t,n,s,l,a)}function Xl(e,t,n,s,a,l){vu(e,t);var o=(t.flags&128)!==0;if(!s&&!o)return a&&Vi(t,n,!1),Gt(e,t,l);s=t.stateNode,om.current=t;var i=o&&typeof n.getDerivedStateFromError!="function"?null:s.render();return t.flags|=1,e!==null&&o?(t.child=an(t,e.child,null,l),t.child=an(t,null,i,l)):We(e,t,i,l),t.memoizedState=s.state,a&&Vi(t,n,!0),t.child}function yu(e){var t=e.stateNode;t.pendingContext?Ui(e,t.pendingContext,t.pendingContext!==t.context):t.context&&Ui(e,t.context,!1),Do(e,t.containerInfo)}function sc(e,t,n,s,a){return sn(),Po(a),t.flags|=256,We(e,t,n,s),t.child}var Yl={dehydrated:null,treeContext:null,retryLane:0};function Kl(e){return{baseLanes:e,cachePool:null,transitions:null}}function ju(e,t,n){var s=t.pendingProps,a=ke.current,l=!1,o=(t.flags&128)!==0,i;if((i=o)||(i=e!==null&&e.memoizedState===null?!1:(a&2)!==0),i?(l=!0,t.flags&=-129):(e===null||e.memoizedState!==null)&&(a|=1),ye(ke,a&1),e===null)return Vl(t),e=t.memoizedState,e!==null&&(e=e.dehydrated,e!==null)?(t.mode&1?e.data==="$!"?t.lanes=8:t.lanes=1073741824:t.lanes=1,null):(o=s.children,e=s.fallback,l?(s=t.mode,l=t.child,o={mode:"hidden",children:o},!(s&1)&&l!==null?(l.childLanes=0,l.pendingProps=o):l=za(o,s,0,null),e=_r(e,s,n,null),l.return=t,e.return=t,l.sibling=e,t.child=l,t.child.memoizedState=Kl(n),t.memoizedState=Yl,e):Go(t,o));if(a=e.memoizedState,a!==null&&(i=a.dehydrated,i!==null))return im(e,t,o,s,i,a,n);if(l){l=s.fallback,o=t.mode,a=e.child,i=a.sibling;var d={mode:"hidden",children:s.children};return!(o&1)&&t.child!==a?(s=t.child,s.childLanes=0,s.pendingProps=d,t.deletions=null):(s=dr(a,d),s.subtreeFlags=a.subtreeFlags&14680064),i!==null?l=dr(i,l):(l=_r(l,o,n,null),l.flags|=2),l.return=t,s.return=t,s.sibling=l,t.child=s,s=l,l=t.child,o=e.child.memoizedState,o=o===null?Kl(n):{baseLanes:o.baseLanes|n,cachePool:null,transitions:o.transitions},l.memoizedState=o,l.childLanes=e.childLanes&~n,t.memoizedState=Yl,s}return l=e.child,e=l.sibling,s=dr(l,{mode:"visible",children:s.children}),!(t.mode&1)&&(s.lanes=n),s.return=t,s.sibling=null,e!==null&&(n=t.deletions,n===null?(t.deletions=[e],t.flags|=16):n.push(e)),t.child=s,t.memoizedState=null,s}function Go(e,t){return t=za({mode:"visible",children:t},e.mode,0,null),t.return=e,e.child=t}function Ts(e,t,n,s){return s!==null&&Po(s),an(t,e.child,null,n),e=Go(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function im(e,t,n,s,a,l,o){if(n)return t.flags&256?(t.flags&=-257,s=ll(Error(U(422))),Ts(e,t,o,s)):t.memoizedState!==null?(t.child=e.child,t.flags|=128,null):(l=s.fallback,a=t.mode,s=za({mode:"visible",children:s.children},a,0,null),l=_r(l,a,o,null),l.flags|=2,s.return=t,l.return=t,s.sibling=l,t.child=s,t.mode&1&&an(t,e.child,null,o),t.child.memoizedState=Kl(o),t.memoizedState=Yl,l);if(!(t.mode&1))return Ts(e,t,o,null);if(a.data==="$!"){if(s=a.nextSibling&&a.nextSibling.dataset,s)var i=s.dgst;return s=i,l=Error(U(419)),s=ll(l,s,void 0),Ts(e,t,o,s)}if(i=(o&e.childLanes)!==0,Ye||i){if(s=Me,s!==null){switch(o&-o){case 4:a=2;break;case 16:a=8;break;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:a=32;break;case 536870912:a=268435456;break;default:a=0}a=a&(s.suspendedLanes|o)?0:a,a!==0&&a!==l.retryLane&&(l.retryLane=a,Wt(e,a),wt(s,e,a,-1))}return qo(),s=ll(Error(U(421))),Ts(e,t,o,s)}return a.data==="$?"?(t.flags|=128,t.child=e.child,t=bm.bind(null,e),a._reactRetry=t,null):(e=l.treeContext,nt=lr(a.nextSibling),st=t,we=!0,jt=null,e!==null&&(ct[dt++]=Dt,ct[dt++]=Ot,ct[dt++]=zr,Dt=e.id,Ot=e.overflow,zr=t),t=Go(t,s.children),t.flags|=4096,t)}function ac(e,t,n){e.lanes|=t;var s=e.alternate;s!==null&&(s.lanes|=t),Bl(e.return,t,n)}function ol(e,t,n,s,a){var l=e.memoizedState;l===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:s,tail:n,tailMode:a}:(l.isBackwards=t,l.rendering=null,l.renderingStartTime=0,l.last=s,l.tail=n,l.tailMode=a)}function bu(e,t,n){var s=t.pendingProps,a=s.revealOrder,l=s.tail;if(We(e,t,s.children,n),s=ke.current,s&2)s=s&1|2,t.flags|=128;else{if(e!==null&&e.flags&128)e:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&ac(e,n,t);else if(e.tag===19)ac(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break e;for(;e.sibling===null;){if(e.return===null||e.return===t)break e;e=e.return}e.sibling.return=e.return,e=e.sibling}s&=1}if(ye(ke,s),!(t.mode&1))t.memoizedState=null;else switch(a){case"forwards":for(n=t.child,a=null;n!==null;)e=n.alternate,e!==null&&ca(e)===null&&(a=n),n=n.sibling;n=a,n===null?(a=t.child,t.child=null):(a=n.sibling,n.sibling=null),ol(t,!1,a,n,l);break;case"backwards":for(n=null,a=t.child,t.child=null;a!==null;){if(e=a.alternate,e!==null&&ca(e)===null){t.child=a;break}e=a.sibling,a.sibling=n,n=a,a=e}ol(t,!0,n,null,l);break;case"together":ol(t,!1,null,null,void 0);break;default:t.memoizedState=null}return t.child}function Bs(e,t){!(t.mode&1)&&e!==null&&(e.alternate=null,t.alternate=null,t.flags|=2)}function Gt(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),Pr|=t.lanes,!(n&t.childLanes))return null;if(e!==null&&t.child!==e.child)throw Error(U(153));if(t.child!==null){for(e=t.child,n=dr(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=dr(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function cm(e,t,n){switch(t.tag){case 3:yu(t),sn();break;case 5:Qd(t);break;case 1:qe(t.type)&&na(t);break;case 4:Do(t,t.stateNode.containerInfo);break;case 10:var s=t.type._context,a=t.memoizedProps.value;ye(la,s._currentValue),s._currentValue=a;break;case 13:if(s=t.memoizedState,s!==null)return s.dehydrated!==null?(ye(ke,ke.current&1),t.flags|=128,null):n&t.child.childLanes?ju(e,t,n):(ye(ke,ke.current&1),e=Gt(e,t,n),e!==null?e.sibling:null);ye(ke,ke.current&1);break;case 19:if(s=(n&t.childLanes)!==0,e.flags&128){if(s)return bu(e,t,n);t.flags|=128}if(a=t.memoizedState,a!==null&&(a.rendering=null,a.tail=null,a.lastEffect=null),ye(ke,ke.current),s)break;return null;case 22:case 23:return t.lanes=0,gu(e,t,n)}return Gt(e,t,n)}var wu,ql,ku,Su;wu=function(e,t){for(var n=t.child;n!==null;){if(n.tag===5||n.tag===6)e.appendChild(n.stateNode);else if(n.tag!==4&&n.child!==null){n.child.return=n,n=n.child;continue}if(n===t)break;for(;n.sibling===null;){if(n.return===null||n.return===t)return;n=n.return}n.sibling.return=n.return,n=n.sibling}};ql=function(){};ku=function(e,t,n,s){var a=e.memoizedProps;if(a!==s){e=t.stateNode,Nr(Pt.current);var l=null;switch(n){case"input":a=yl(e,a),s=yl(e,s),l=[];break;case"select":a=Ne({},a,{value:void 0}),s=Ne({},s,{value:void 0}),l=[];break;case"textarea":a=wl(e,a),s=wl(e,s),l=[];break;default:typeof a.onClick!="function"&&typeof s.onClick=="function"&&(e.onclick=ta)}Sl(n,s);var o;n=null;for(u in a)if(!s.hasOwnProperty(u)&&a.hasOwnProperty(u)&&a[u]!=null)if(u==="style"){var i=a[u];for(o in i)i.hasOwnProperty(o)&&(n||(n={}),n[o]="")}else u!=="dangerouslySetInnerHTML"&&u!=="children"&&u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&u!=="autoFocus"&&(Bn.hasOwnProperty(u)?l||(l=[]):(l=l||[]).push(u,null));for(u in s){var d=s[u];if(i=a!=null?a[u]:void 0,s.hasOwnProperty(u)&&d!==i&&(d!=null||i!=null))if(u==="style")if(i){for(o in i)!i.hasOwnProperty(o)||d&&d.hasOwnProperty(o)||(n||(n={}),n[o]="");for(o in d)d.hasOwnProperty(o)&&i[o]!==d[o]&&(n||(n={}),n[o]=d[o])}else n||(l||(l=[]),l.push(u,n)),n=d;else u==="dangerouslySetInnerHTML"?(d=d?d.__html:void 0,i=i?i.__html:void 0,d!=null&&i!==d&&(l=l||[]).push(u,d)):u==="children"?typeof d!="string"&&typeof d!="number"||(l=l||[]).push(u,""+d):u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&(Bn.hasOwnProperty(u)?(d!=null&&u==="onScroll"&&je("scroll",e),l||i===d||(l=[])):(l=l||[]).push(u,d))}n&&(l=l||[]).push("style",n);var u=l;(t.updateQueue=u)&&(t.flags|=4)}};Su=function(e,t,n,s){n!==s&&(t.flags|=4)};function Cn(e,t){if(!we)switch(e.tailMode){case"hidden":t=e.tail;for(var n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null;break;case"collapsed":n=e.tail;for(var s=null;n!==null;)n.alternate!==null&&(s=n),n=n.sibling;s===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:s.sibling=null}}function $e(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,s=0;if(t)for(var a=e.child;a!==null;)n|=a.lanes|a.childLanes,s|=a.subtreeFlags&14680064,s|=a.flags&14680064,a.return=e,a=a.sibling;else for(a=e.child;a!==null;)n|=a.lanes|a.childLanes,s|=a.subtreeFlags,s|=a.flags,a.return=e,a=a.sibling;return e.subtreeFlags|=s,e.childLanes=n,t}function dm(e,t,n){var s=t.pendingProps;switch(To(t),t.tag){case 2:case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return $e(t),null;case 1:return qe(t.type)&&ra(),$e(t),null;case 3:return s=t.stateNode,ln(),be(Ke),be(Ve),Ao(),s.pendingContext&&(s.context=s.pendingContext,s.pendingContext=null),(e===null||e.child===null)&&(Es(t)?t.flags|=4:e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,jt!==null&&(ao(jt),jt=null))),ql(e,t),$e(t),null;case 5:Oo(t);var a=Nr(ts.current);if(n=t.type,e!==null&&t.stateNode!=null)ku(e,t,n,s,a),e.ref!==t.ref&&(t.flags|=512,t.flags|=2097152);else{if(!s){if(t.stateNode===null)throw Error(U(166));return $e(t),null}if(e=Nr(Pt.current),Es(t)){s=t.stateNode,n=t.type;var l=t.memoizedProps;switch(s[zt]=t,s[Zn]=l,e=(t.mode&1)!==0,n){case"dialog":je("cancel",s),je("close",s);break;case"iframe":case"object":case"embed":je("load",s);break;case"video":case"audio":for(a=0;a<Mn.length;a++)je(Mn[a],s);break;case"source":je("error",s);break;case"img":case"image":case"link":je("error",s),je("load",s);break;case"details":je("toggle",s);break;case"input":mi(s,l),je("invalid",s);break;case"select":s._wrapperState={wasMultiple:!!l.multiple},je("invalid",s);break;case"textarea":xi(s,l),je("invalid",s)}Sl(n,l),a=null;for(var o in l)if(l.hasOwnProperty(o)){var i=l[o];o==="children"?typeof i=="string"?s.textContent!==i&&(l.suppressHydrationWarning!==!0&&_s(s.textContent,i,e),a=["children",i]):typeof i=="number"&&s.textContent!==""+i&&(l.suppressHydrationWarning!==!0&&_s(s.textContent,i,e),a=["children",""+i]):Bn.hasOwnProperty(o)&&i!=null&&o==="onScroll"&&je("scroll",s)}switch(n){case"input":ys(s),hi(s,l,!0);break;case"textarea":ys(s),gi(s);break;case"select":case"option":break;default:typeof l.onClick=="function"&&(s.onclick=ta)}s=a,t.updateQueue=s,s!==null&&(t.flags|=4)}else{o=a.nodeType===9?a:a.ownerDocument,e==="http://www.w3.org/1999/xhtml"&&(e=Jc(n)),e==="http://www.w3.org/1999/xhtml"?n==="script"?(e=o.createElement("div"),e.innerHTML="<script><\/script>",e=e.removeChild(e.firstChild)):typeof s.is=="string"?e=o.createElement(n,{is:s.is}):(e=o.createElement(n),n==="select"&&(o=e,s.multiple?o.multiple=!0:s.size&&(o.size=s.size))):e=o.createElementNS(e,n),e[zt]=t,e[Zn]=s,wu(e,t,!1,!1),t.stateNode=e;e:{switch(o=Nl(n,s),n){case"dialog":je("cancel",e),je("close",e),a=s;break;case"iframe":case"object":case"embed":je("load",e),a=s;break;case"video":case"audio":for(a=0;a<Mn.length;a++)je(Mn[a],e);a=s;break;case"source":je("error",e),a=s;break;case"img":case"image":case"link":je("error",e),je("load",e),a=s;break;case"details":je("toggle",e),a=s;break;case"input":mi(e,s),a=yl(e,s),je("invalid",e);break;case"option":a=s;break;case"select":e._wrapperState={wasMultiple:!!s.multiple},a=Ne({},s,{value:void 0}),je("invalid",e);break;case"textarea":xi(e,s),a=wl(e,s),je("invalid",e);break;default:a=s}Sl(n,a),i=a;for(l in i)if(i.hasOwnProperty(l)){var d=i[l];l==="style"?td(e,d):l==="dangerouslySetInnerHTML"?(d=d?d.__html:void 0,d!=null&&Zc(e,d)):l==="children"?typeof d=="string"?(n!=="textarea"||d!=="")&&Wn(e,d):typeof d=="number"&&Wn(e,""+d):l!=="suppressContentEditableWarning"&&l!=="suppressHydrationWarning"&&l!=="autoFocus"&&(Bn.hasOwnProperty(l)?d!=null&&l==="onScroll"&&je("scroll",e):d!=null&&ho(e,l,d,o))}switch(n){case"input":ys(e),hi(e,s,!1);break;case"textarea":ys(e),gi(e);break;case"option":s.value!=null&&e.setAttribute("value",""+ur(s.value));break;case"select":e.multiple=!!s.multiple,l=s.value,l!=null?Kr(e,!!s.multiple,l,!1):s.defaultValue!=null&&Kr(e,!!s.multiple,s.defaultValue,!0);break;default:typeof a.onClick=="function"&&(e.onclick=ta)}switch(n){case"button":case"input":case"select":case"textarea":s=!!s.autoFocus;break e;case"img":s=!0;break e;default:s=!1}}s&&(t.flags|=4)}t.ref!==null&&(t.flags|=512,t.flags|=2097152)}return $e(t),null;case 6:if(e&&t.stateNode!=null)Su(e,t,e.memoizedProps,s);else{if(typeof s!="string"&&t.stateNode===null)throw Error(U(166));if(n=Nr(ts.current),Nr(Pt.current),Es(t)){if(s=t.stateNode,n=t.memoizedProps,s[zt]=t,(l=s.nodeValue!==n)&&(e=st,e!==null))switch(e.tag){case 3:_s(s.nodeValue,n,(e.mode&1)!==0);break;case 5:e.memoizedProps.suppressHydrationWarning!==!0&&_s(s.nodeValue,n,(e.mode&1)!==0)}l&&(t.flags|=4)}else s=(n.nodeType===9?n:n.ownerDocument).createTextNode(s),s[zt]=t,t.stateNode=s}return $e(t),null;case 13:if(be(ke),s=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(we&&nt!==null&&t.mode&1&&!(t.flags&128))Vd(),sn(),t.flags|=98560,l=!1;else if(l=Es(t),s!==null&&s.dehydrated!==null){if(e===null){if(!l)throw Error(U(318));if(l=t.memoizedState,l=l!==null?l.dehydrated:null,!l)throw Error(U(317));l[zt]=t}else sn(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;$e(t),l=!1}else jt!==null&&(ao(jt),jt=null),l=!0;if(!l)return t.flags&65536?t:null}return t.flags&128?(t.lanes=n,t):(s=s!==null,s!==(e!==null&&e.memoizedState!==null)&&s&&(t.child.flags|=8192,t.mode&1&&(e===null||ke.current&1?Pe===0&&(Pe=3):qo())),t.updateQueue!==null&&(t.flags|=4),$e(t),null);case 4:return ln(),ql(e,t),e===null&&qn(t.stateNode.containerInfo),$e(t),null;case 10:return Ro(t.type._context),$e(t),null;case 17:return qe(t.type)&&ra(),$e(t),null;case 19:if(be(ke),l=t.memoizedState,l===null)return $e(t),null;if(s=(t.flags&128)!==0,o=l.rendering,o===null)if(s)Cn(l,!1);else{if(Pe!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(o=ca(e),o!==null){for(t.flags|=128,Cn(l,!1),s=o.updateQueue,s!==null&&(t.updateQueue=s,t.flags|=4),t.subtreeFlags=0,s=n,n=t.child;n!==null;)l=n,e=s,l.flags&=14680066,o=l.alternate,o===null?(l.childLanes=0,l.lanes=e,l.child=null,l.subtreeFlags=0,l.memoizedProps=null,l.memoizedState=null,l.updateQueue=null,l.dependencies=null,l.stateNode=null):(l.childLanes=o.childLanes,l.lanes=o.lanes,l.child=o.child,l.subtreeFlags=0,l.deletions=null,l.memoizedProps=o.memoizedProps,l.memoizedState=o.memoizedState,l.updateQueue=o.updateQueue,l.type=o.type,e=o.dependencies,l.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),n=n.sibling;return ye(ke,ke.current&1|2),t.child}e=e.sibling}l.tail!==null&&_e()>cn&&(t.flags|=128,s=!0,Cn(l,!1),t.lanes=4194304)}else{if(!s)if(e=ca(o),e!==null){if(t.flags|=128,s=!0,n=e.updateQueue,n!==null&&(t.updateQueue=n,t.flags|=4),Cn(l,!0),l.tail===null&&l.tailMode==="hidden"&&!o.alternate&&!we)return $e(t),null}else 2*_e()-l.renderingStartTime>cn&&n!==1073741824&&(t.flags|=128,s=!0,Cn(l,!1),t.lanes=4194304);l.isBackwards?(o.sibling=t.child,t.child=o):(n=l.last,n!==null?n.sibling=o:t.child=o,l.last=o)}return l.tail!==null?(t=l.tail,l.rendering=t,l.tail=t.sibling,l.renderingStartTime=_e(),t.sibling=null,n=ke.current,ye(ke,s?n&1|2:n&1),t):($e(t),null);case 22:case 23:return Ko(),s=t.memoizedState!==null,e!==null&&e.memoizedState!==null!==s&&(t.flags|=8192),s&&t.mode&1?rt&1073741824&&($e(t),t.subtreeFlags&6&&(t.flags|=8192)):$e(t),null;case 24:return null;case 25:return null}throw Error(U(156,t.tag))}function um(e,t){switch(To(t),t.tag){case 1:return qe(t.type)&&ra(),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return ln(),be(Ke),be(Ve),Ao(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 5:return Oo(t),null;case 13:if(be(ke),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(U(340));sn()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return be(ke),null;case 4:return ln(),null;case 10:return Ro(t.type._context),null;case 22:case 23:return Ko(),null;case 24:return null;default:return null}}var Ps=!1,Ue=!1,pm=typeof WeakSet=="function"?WeakSet:Set,q=null;function Xr(e,t){var n=e.ref;if(n!==null)if(typeof n=="function")try{n(null)}catch(s){Ce(e,t,s)}else n.current=null}function Jl(e,t,n){try{n()}catch(s){Ce(e,t,s)}}var lc=!1;function fm(e,t){if(Ll=Js,e=zd(),Eo(e)){if("selectionStart"in e)var n={start:e.selectionStart,end:e.selectionEnd};else e:{n=(n=e.ownerDocument)&&n.defaultView||window;var s=n.getSelection&&n.getSelection();if(s&&s.rangeCount!==0){n=s.anchorNode;var a=s.anchorOffset,l=s.focusNode;s=s.focusOffset;try{n.nodeType,l.nodeType}catch{n=null;break e}var o=0,i=-1,d=-1,u=0,y=0,g=e,x=null;t:for(;;){for(var k;g!==n||a!==0&&g.nodeType!==3||(i=o+a),g!==l||s!==0&&g.nodeType!==3||(d=o+s),g.nodeType===3&&(o+=g.nodeValue.length),(k=g.firstChild)!==null;)x=g,g=k;for(;;){if(g===e)break t;if(x===n&&++u===a&&(i=o),x===l&&++y===s&&(d=o),(k=g.nextSibling)!==null)break;g=x,x=g.parentNode}g=k}n=i===-1||d===-1?null:{start:i,end:d}}else n=null}n=n||{start:0,end:0}}else n=null;for(Fl={focusedElem:e,selectionRange:n},Js=!1,q=t;q!==null;)if(t=q,e=t.child,(t.subtreeFlags&1028)!==0&&e!==null)e.return=t,q=e;else for(;q!==null;){t=q;try{var S=t.alternate;if(t.flags&1024)switch(t.tag){case 0:case 11:case 15:break;case 1:if(S!==null){var z=S.memoizedProps,R=S.memoizedState,f=t.stateNode,p=f.getSnapshotBeforeUpdate(t.elementType===t.type?z:vt(t.type,z),R);f.__reactInternalSnapshotBeforeUpdate=p}break;case 3:var m=t.stateNode.containerInfo;m.nodeType===1?m.textContent="":m.nodeType===9&&m.documentElement&&m.removeChild(m.documentElement);break;case 5:case 6:case 4:case 17:break;default:throw Error(U(163))}}catch(h){Ce(t,t.return,h)}if(e=t.sibling,e!==null){e.return=t.return,q=e;break}q=t.return}return S=lc,lc=!1,S}function $n(e,t,n){var s=t.updateQueue;if(s=s!==null?s.lastEffect:null,s!==null){var a=s=s.next;do{if((a.tag&e)===e){var l=a.destroy;a.destroy=void 0,l!==void 0&&Jl(t,n,l)}a=a.next}while(a!==s)}}function _a(e,t){if(t=t.updateQueue,t=t!==null?t.lastEffect:null,t!==null){var n=t=t.next;do{if((n.tag&e)===e){var s=n.create;n.destroy=s()}n=n.next}while(n!==t)}}function Zl(e){var t=e.ref;if(t!==null){var n=e.stateNode;switch(e.tag){case 5:e=n;break;default:e=n}typeof t=="function"?t(e):t.current=e}}function Nu(e){var t=e.alternate;t!==null&&(e.alternate=null,Nu(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&(delete t[zt],delete t[Zn],delete t[Al],delete t[Yf],delete t[Kf])),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}function Cu(e){return e.tag===5||e.tag===3||e.tag===4}function oc(e){e:for(;;){for(;e.sibling===null;){if(e.return===null||Cu(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.flags&2||e.child===null||e.tag===4)continue e;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function eo(e,t,n){var s=e.tag;if(s===5||s===6)e=e.stateNode,t?n.nodeType===8?n.parentNode.insertBefore(e,t):n.insertBefore(e,t):(n.nodeType===8?(t=n.parentNode,t.insertBefore(e,n)):(t=n,t.appendChild(e)),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=ta));else if(s!==4&&(e=e.child,e!==null))for(eo(e,t,n),e=e.sibling;e!==null;)eo(e,t,n),e=e.sibling}function to(e,t,n){var s=e.tag;if(s===5||s===6)e=e.stateNode,t?n.insertBefore(e,t):n.appendChild(e);else if(s!==4&&(e=e.child,e!==null))for(to(e,t,n),e=e.sibling;e!==null;)to(e,t,n),e=e.sibling}var Le=null,yt=!1;function Kt(e,t,n){for(n=n.child;n!==null;)_u(e,t,n),n=n.sibling}function _u(e,t,n){if(Tt&&typeof Tt.onCommitFiberUnmount=="function")try{Tt.onCommitFiberUnmount(ya,n)}catch{}switch(n.tag){case 5:Ue||Xr(n,t);case 6:var s=Le,a=yt;Le=null,Kt(e,t,n),Le=s,yt=a,Le!==null&&(yt?(e=Le,n=n.stateNode,e.nodeType===8?e.parentNode.removeChild(n):e.removeChild(n)):Le.removeChild(n.stateNode));break;case 18:Le!==null&&(yt?(e=Le,n=n.stateNode,e.nodeType===8?el(e.parentNode,n):e.nodeType===1&&el(e,n),Xn(e)):el(Le,n.stateNode));break;case 4:s=Le,a=yt,Le=n.stateNode.containerInfo,yt=!0,Kt(e,t,n),Le=s,yt=a;break;case 0:case 11:case 14:case 15:if(!Ue&&(s=n.updateQueue,s!==null&&(s=s.lastEffect,s!==null))){a=s=s.next;do{var l=a,o=l.destroy;l=l.tag,o!==void 0&&(l&2||l&4)&&Jl(n,t,o),a=a.next}while(a!==s)}Kt(e,t,n);break;case 1:if(!Ue&&(Xr(n,t),s=n.stateNode,typeof s.componentWillUnmount=="function"))try{s.props=n.memoizedProps,s.state=n.memoizedState,s.componentWillUnmount()}catch(i){Ce(n,t,i)}Kt(e,t,n);break;case 21:Kt(e,t,n);break;case 22:n.mode&1?(Ue=(s=Ue)||n.memoizedState!==null,Kt(e,t,n),Ue=s):Kt(e,t,n);break;default:Kt(e,t,n)}}function ic(e){var t=e.updateQueue;if(t!==null){e.updateQueue=null;var n=e.stateNode;n===null&&(n=e.stateNode=new pm),t.forEach(function(s){var a=wm.bind(null,e,s);n.has(s)||(n.add(s),s.then(a,a))})}}function gt(e,t){var n=t.deletions;if(n!==null)for(var s=0;s<n.length;s++){var a=n[s];try{var l=e,o=t,i=o;e:for(;i!==null;){switch(i.tag){case 5:Le=i.stateNode,yt=!1;break e;case 3:Le=i.stateNode.containerInfo,yt=!0;break e;case 4:Le=i.stateNode.containerInfo,yt=!0;break e}i=i.return}if(Le===null)throw Error(U(160));_u(l,o,a),Le=null,yt=!1;var d=a.alternate;d!==null&&(d.return=null),a.return=null}catch(u){Ce(a,t,u)}}if(t.subtreeFlags&12854)for(t=t.child;t!==null;)Eu(t,e),t=t.sibling}function Eu(e,t){var n=e.alternate,s=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(gt(t,e),_t(e),s&4){try{$n(3,e,e.return),_a(3,e)}catch(z){Ce(e,e.return,z)}try{$n(5,e,e.return)}catch(z){Ce(e,e.return,z)}}break;case 1:gt(t,e),_t(e),s&512&&n!==null&&Xr(n,n.return);break;case 5:if(gt(t,e),_t(e),s&512&&n!==null&&Xr(n,n.return),e.flags&32){var a=e.stateNode;try{Wn(a,"")}catch(z){Ce(e,e.return,z)}}if(s&4&&(a=e.stateNode,a!=null)){var l=e.memoizedProps,o=n!==null?n.memoizedProps:l,i=e.type,d=e.updateQueue;if(e.updateQueue=null,d!==null)try{i==="input"&&l.type==="radio"&&l.name!=null&&Kc(a,l),Nl(i,o);var u=Nl(i,l);for(o=0;o<d.length;o+=2){var y=d[o],g=d[o+1];y==="style"?td(a,g):y==="dangerouslySetInnerHTML"?Zc(a,g):y==="children"?Wn(a,g):ho(a,y,g,u)}switch(i){case"input":jl(a,l);break;case"textarea":qc(a,l);break;case"select":var x=a._wrapperState.wasMultiple;a._wrapperState.wasMultiple=!!l.multiple;var k=l.value;k!=null?Kr(a,!!l.multiple,k,!1):x!==!!l.multiple&&(l.defaultValue!=null?Kr(a,!!l.multiple,l.defaultValue,!0):Kr(a,!!l.multiple,l.multiple?[]:"",!1))}a[Zn]=l}catch(z){Ce(e,e.return,z)}}break;case 6:if(gt(t,e),_t(e),s&4){if(e.stateNode===null)throw Error(U(162));a=e.stateNode,l=e.memoizedProps;try{a.nodeValue=l}catch(z){Ce(e,e.return,z)}}break;case 3:if(gt(t,e),_t(e),s&4&&n!==null&&n.memoizedState.isDehydrated)try{Xn(t.containerInfo)}catch(z){Ce(e,e.return,z)}break;case 4:gt(t,e),_t(e);break;case 13:gt(t,e),_t(e),a=e.child,a.flags&8192&&(l=a.memoizedState!==null,a.stateNode.isHidden=l,!l||a.alternate!==null&&a.alternate.memoizedState!==null||(Xo=_e())),s&4&&ic(e);break;case 22:if(y=n!==null&&n.memoizedState!==null,e.mode&1?(Ue=(u=Ue)||y,gt(t,e),Ue=u):gt(t,e),_t(e),s&8192){if(u=e.memoizedState!==null,(e.stateNode.isHidden=u)&&!y&&e.mode&1)for(q=e,y=e.child;y!==null;){for(g=q=y;q!==null;){switch(x=q,k=x.child,x.tag){case 0:case 11:case 14:case 15:$n(4,x,x.return);break;case 1:Xr(x,x.return);var S=x.stateNode;if(typeof S.componentWillUnmount=="function"){s=x,n=x.return;try{t=s,S.props=t.memoizedProps,S.state=t.memoizedState,S.componentWillUnmount()}catch(z){Ce(s,n,z)}}break;case 5:Xr(x,x.return);break;case 22:if(x.memoizedState!==null){dc(g);continue}}k!==null?(k.return=x,q=k):dc(g)}y=y.sibling}e:for(y=null,g=e;;){if(g.tag===5){if(y===null){y=g;try{a=g.stateNode,u?(l=a.style,typeof l.setProperty=="function"?l.setProperty("display","none","important"):l.display="none"):(i=g.stateNode,d=g.memoizedProps.style,o=d!=null&&d.hasOwnProperty("display")?d.display:null,i.style.display=ed("display",o))}catch(z){Ce(e,e.return,z)}}}else if(g.tag===6){if(y===null)try{g.stateNode.nodeValue=u?"":g.memoizedProps}catch(z){Ce(e,e.return,z)}}else if((g.tag!==22&&g.tag!==23||g.memoizedState===null||g===e)&&g.child!==null){g.child.return=g,g=g.child;continue}if(g===e)break e;for(;g.sibling===null;){if(g.return===null||g.return===e)break e;y===g&&(y=null),g=g.return}y===g&&(y=null),g.sibling.return=g.return,g=g.sibling}}break;case 19:gt(t,e),_t(e),s&4&&ic(e);break;case 21:break;default:gt(t,e),_t(e)}}function _t(e){var t=e.flags;if(t&2){try{e:{for(var n=e.return;n!==null;){if(Cu(n)){var s=n;break e}n=n.return}throw Error(U(160))}switch(s.tag){case 5:var a=s.stateNode;s.flags&32&&(Wn(a,""),s.flags&=-33);var l=oc(e);to(e,l,a);break;case 3:case 4:var o=s.stateNode.containerInfo,i=oc(e);eo(e,i,o);break;default:throw Error(U(161))}}catch(d){Ce(e,e.return,d)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function mm(e,t,n){q=e,zu(e)}function zu(e,t,n){for(var s=(e.mode&1)!==0;q!==null;){var a=q,l=a.child;if(a.tag===22&&s){var o=a.memoizedState!==null||Ps;if(!o){var i=a.alternate,d=i!==null&&i.memoizedState!==null||Ue;i=Ps;var u=Ue;if(Ps=o,(Ue=d)&&!u)for(q=a;q!==null;)o=q,d=o.child,o.tag===22&&o.memoizedState!==null?uc(a):d!==null?(d.return=o,q=d):uc(a);for(;l!==null;)q=l,zu(l),l=l.sibling;q=a,Ps=i,Ue=u}cc(e)}else a.subtreeFlags&8772&&l!==null?(l.return=a,q=l):cc(e)}}function cc(e){for(;q!==null;){var t=q;if(t.flags&8772){var n=t.alternate;try{if(t.flags&8772)switch(t.tag){case 0:case 11:case 15:Ue||_a(5,t);break;case 1:var s=t.stateNode;if(t.flags&4&&!Ue)if(n===null)s.componentDidMount();else{var a=t.elementType===t.type?n.memoizedProps:vt(t.type,n.memoizedProps);s.componentDidUpdate(a,n.memoizedState,s.__reactInternalSnapshotBeforeUpdate)}var l=t.updateQueue;l!==null&&Qi(t,l,s);break;case 3:var o=t.updateQueue;if(o!==null){if(n=null,t.child!==null)switch(t.child.tag){case 5:n=t.child.stateNode;break;case 1:n=t.child.stateNode}Qi(t,o,n)}break;case 5:var i=t.stateNode;if(n===null&&t.flags&4){n=i;var d=t.memoizedProps;switch(t.type){case"button":case"input":case"select":case"textarea":d.autoFocus&&n.focus();break;case"img":d.src&&(n.src=d.src)}}break;case 6:break;case 4:break;case 12:break;case 13:if(t.memoizedState===null){var u=t.alternate;if(u!==null){var y=u.memoizedState;if(y!==null){var g=y.dehydrated;g!==null&&Xn(g)}}}break;case 19:case 17:case 21:case 22:case 23:case 25:break;default:throw Error(U(163))}Ue||t.flags&512&&Zl(t)}catch(x){Ce(t,t.return,x)}}if(t===e){q=null;break}if(n=t.sibling,n!==null){n.return=t.return,q=n;break}q=t.return}}function dc(e){for(;q!==null;){var t=q;if(t===e){q=null;break}var n=t.sibling;if(n!==null){n.return=t.return,q=n;break}q=t.return}}function uc(e){for(;q!==null;){var t=q;try{switch(t.tag){case 0:case 11:case 15:var n=t.return;try{_a(4,t)}catch(d){Ce(t,n,d)}break;case 1:var s=t.stateNode;if(typeof s.componentDidMount=="function"){var a=t.return;try{s.componentDidMount()}catch(d){Ce(t,a,d)}}var l=t.return;try{Zl(t)}catch(d){Ce(t,l,d)}break;case 5:var o=t.return;try{Zl(t)}catch(d){Ce(t,o,d)}}}catch(d){Ce(t,t.return,d)}if(t===e){q=null;break}var i=t.sibling;if(i!==null){i.return=t.return,q=i;break}q=t.return}}var hm=Math.ceil,pa=Qt.ReactCurrentDispatcher,Ho=Qt.ReactCurrentOwner,pt=Qt.ReactCurrentBatchConfig,fe=0,Me=null,ze=null,Fe=0,rt=0,Yr=gr(0),Pe=0,as=null,Pr=0,Ea=0,Qo=0,Un=null,Xe=null,Xo=0,cn=1/0,Lt=null,fa=!1,ro=null,ir=null,Is=!1,rr=null,ma=0,Vn=0,no=null,Ws=-1,Gs=0;function Ge(){return fe&6?_e():Ws!==-1?Ws:Ws=_e()}function cr(e){return e.mode&1?fe&2&&Fe!==0?Fe&-Fe:Jf.transition!==null?(Gs===0&&(Gs=fd()),Gs):(e=ge,e!==0||(e=window.event,e=e===void 0?16:jd(e.type)),e):1}function wt(e,t,n,s){if(50<Vn)throw Vn=0,no=null,Error(U(185));cs(e,n,s),(!(fe&2)||e!==Me)&&(e===Me&&(!(fe&2)&&(Ea|=n),Pe===4&&er(e,Fe)),Je(e,s),n===1&&fe===0&&!(t.mode&1)&&(cn=_e()+500,Sa&&vr()))}function Je(e,t){var n=e.callbackNode;qp(e,t);var s=qs(e,e===Me?Fe:0);if(s===0)n!==null&&ji(n),e.callbackNode=null,e.callbackPriority=0;else if(t=s&-s,e.callbackPriority!==t){if(n!=null&&ji(n),t===1)e.tag===0?qf(pc.bind(null,e)):Ad(pc.bind(null,e)),Qf(function(){!(fe&6)&&vr()}),n=null;else{switch(md(s)){case 1:n=jo;break;case 4:n=ud;break;case 16:n=Ks;break;case 536870912:n=pd;break;default:n=Ks}n=Du(n,Tu.bind(null,e))}e.callbackPriority=t,e.callbackNode=n}}function Tu(e,t){if(Ws=-1,Gs=0,fe&6)throw Error(U(327));var n=e.callbackNode;if(tn()&&e.callbackNode!==n)return null;var s=qs(e,e===Me?Fe:0);if(s===0)return null;if(s&30||s&e.expiredLanes||t)t=ha(e,s);else{t=s;var a=fe;fe|=2;var l=Iu();(Me!==e||Fe!==t)&&(Lt=null,cn=_e()+500,Cr(e,t));do try{vm();break}catch(i){Pu(e,i)}while(!0);Mo(),pa.current=l,fe=a,ze!==null?t=0:(Me=null,Fe=0,t=Pe)}if(t!==0){if(t===2&&(a=Tl(e),a!==0&&(s=a,t=so(e,a))),t===1)throw n=as,Cr(e,0),er(e,s),Je(e,_e()),n;if(t===6)er(e,s);else{if(a=e.current.alternate,!(s&30)&&!xm(a)&&(t=ha(e,s),t===2&&(l=Tl(e),l!==0&&(s=l,t=so(e,l))),t===1))throw n=as,Cr(e,0),er(e,s),Je(e,_e()),n;switch(e.finishedWork=a,e.finishedLanes=s,t){case 0:case 1:throw Error(U(345));case 2:wr(e,Xe,Lt);break;case 3:if(er(e,s),(s&130023424)===s&&(t=Xo+500-_e(),10<t)){if(qs(e,0)!==0)break;if(a=e.suspendedLanes,(a&s)!==s){Ge(),e.pingedLanes|=e.suspendedLanes&a;break}e.timeoutHandle=Ol(wr.bind(null,e,Xe,Lt),t);break}wr(e,Xe,Lt);break;case 4:if(er(e,s),(s&4194240)===s)break;for(t=e.eventTimes,a=-1;0<s;){var o=31-bt(s);l=1<<o,o=t[o],o>a&&(a=o),s&=~l}if(s=a,s=_e()-s,s=(120>s?120:480>s?480:1080>s?1080:1920>s?1920:3e3>s?3e3:4320>s?4320:1960*hm(s/1960))-s,10<s){e.timeoutHandle=Ol(wr.bind(null,e,Xe,Lt),s);break}wr(e,Xe,Lt);break;case 5:wr(e,Xe,Lt);break;default:throw Error(U(329))}}}return Je(e,_e()),e.callbackNode===n?Tu.bind(null,e):null}function so(e,t){var n=Un;return e.current.memoizedState.isDehydrated&&(Cr(e,t).flags|=256),e=ha(e,t),e!==2&&(t=Xe,Xe=n,t!==null&&ao(t)),e}function ao(e){Xe===null?Xe=e:Xe.push.apply(Xe,e)}function xm(e){for(var t=e;;){if(t.flags&16384){var n=t.updateQueue;if(n!==null&&(n=n.stores,n!==null))for(var s=0;s<n.length;s++){var a=n[s],l=a.getSnapshot;a=a.value;try{if(!kt(l(),a))return!1}catch{return!1}}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function er(e,t){for(t&=~Qo,t&=~Ea,e.suspendedLanes|=t,e.pingedLanes&=~t,e=e.expirationTimes;0<t;){var n=31-bt(t),s=1<<n;e[n]=-1,t&=~s}}function pc(e){if(fe&6)throw Error(U(327));tn();var t=qs(e,0);if(!(t&1))return Je(e,_e()),null;var n=ha(e,t);if(e.tag!==0&&n===2){var s=Tl(e);s!==0&&(t=s,n=so(e,s))}if(n===1)throw n=as,Cr(e,0),er(e,t),Je(e,_e()),n;if(n===6)throw Error(U(345));return e.finishedWork=e.current.alternate,e.finishedLanes=t,wr(e,Xe,Lt),Je(e,_e()),null}function Yo(e,t){var n=fe;fe|=1;try{return e(t)}finally{fe=n,fe===0&&(cn=_e()+500,Sa&&vr())}}function Ir(e){rr!==null&&rr.tag===0&&!(fe&6)&&tn();var t=fe;fe|=1;var n=pt.transition,s=ge;try{if(pt.transition=null,ge=1,e)return e()}finally{ge=s,pt.transition=n,fe=t,!(fe&6)&&vr()}}function Ko(){rt=Yr.current,be(Yr)}function Cr(e,t){e.finishedWork=null,e.finishedLanes=0;var n=e.timeoutHandle;if(n!==-1&&(e.timeoutHandle=-1,Hf(n)),ze!==null)for(n=ze.return;n!==null;){var s=n;switch(To(s),s.tag){case 1:s=s.type.childContextTypes,s!=null&&ra();break;case 3:ln(),be(Ke),be(Ve),Ao();break;case 5:Oo(s);break;case 4:ln();break;case 13:be(ke);break;case 19:be(ke);break;case 10:Ro(s.type._context);break;case 22:case 23:Ko()}n=n.return}if(Me=e,ze=e=dr(e.current,null),Fe=rt=t,Pe=0,as=null,Qo=Ea=Pr=0,Xe=Un=null,Sr!==null){for(t=0;t<Sr.length;t++)if(n=Sr[t],s=n.interleaved,s!==null){n.interleaved=null;var a=s.next,l=n.pending;if(l!==null){var o=l.next;l.next=a,s.next=o}n.pending=s}Sr=null}return e}function Pu(e,t){do{var n=ze;try{if(Mo(),Us.current=ua,da){for(var s=Se.memoizedState;s!==null;){var a=s.queue;a!==null&&(a.pending=null),s=s.next}da=!1}if(Tr=0,Ie=Te=Se=null,An=!1,rs=0,Ho.current=null,n===null||n.return===null){Pe=1,as=t,ze=null;break}e:{var l=e,o=n.return,i=n,d=t;if(t=Fe,i.flags|=32768,d!==null&&typeof d=="object"&&typeof d.then=="function"){var u=d,y=i,g=y.tag;if(!(y.mode&1)&&(g===0||g===11||g===15)){var x=y.alternate;x?(y.updateQueue=x.updateQueue,y.memoizedState=x.memoizedState,y.lanes=x.lanes):(y.updateQueue=null,y.memoizedState=null)}var k=Zi(o);if(k!==null){k.flags&=-257,ec(k,o,i,l,t),k.mode&1&&Ji(l,u,t),t=k,d=u;var S=t.updateQueue;if(S===null){var z=new Set;z.add(d),t.updateQueue=z}else S.add(d);break e}else{if(!(t&1)){Ji(l,u,t),qo();break e}d=Error(U(426))}}else if(we&&i.mode&1){var R=Zi(o);if(R!==null){!(R.flags&65536)&&(R.flags|=256),ec(R,o,i,l,t),Po(on(d,i));break e}}l=d=on(d,i),Pe!==4&&(Pe=2),Un===null?Un=[l]:Un.push(l),l=o;do{switch(l.tag){case 3:l.flags|=65536,t&=-t,l.lanes|=t;var f=mu(l,d,t);Hi(l,f);break e;case 1:i=d;var p=l.type,m=l.stateNode;if(!(l.flags&128)&&(typeof p.getDerivedStateFromError=="function"||m!==null&&typeof m.componentDidCatch=="function"&&(ir===null||!ir.has(m)))){l.flags|=65536,t&=-t,l.lanes|=t;var h=hu(l,i,t);Hi(l,h);break e}}l=l.return}while(l!==null)}Ru(n)}catch(j){t=j,ze===n&&n!==null&&(ze=n=n.return);continue}break}while(!0)}function Iu(){var e=pa.current;return pa.current=ua,e===null?ua:e}function qo(){(Pe===0||Pe===3||Pe===2)&&(Pe=4),Me===null||!(Pr&268435455)&&!(Ea&268435455)||er(Me,Fe)}function ha(e,t){var n=fe;fe|=2;var s=Iu();(Me!==e||Fe!==t)&&(Lt=null,Cr(e,t));do try{gm();break}catch(a){Pu(e,a)}while(!0);if(Mo(),fe=n,pa.current=s,ze!==null)throw Error(U(261));return Me=null,Fe=0,Pe}function gm(){for(;ze!==null;)Mu(ze)}function vm(){for(;ze!==null&&!Vp();)Mu(ze)}function Mu(e){var t=Fu(e.alternate,e,rt);e.memoizedProps=e.pendingProps,t===null?Ru(e):ze=t,Ho.current=null}function Ru(e){var t=e;do{var n=t.alternate;if(e=t.return,t.flags&32768){if(n=um(n,t),n!==null){n.flags&=32767,ze=n;return}if(e!==null)e.flags|=32768,e.subtreeFlags=0,e.deletions=null;else{Pe=6,ze=null;return}}else if(n=dm(n,t,rt),n!==null){ze=n;return}if(t=t.sibling,t!==null){ze=t;return}ze=t=e}while(t!==null);Pe===0&&(Pe=5)}function wr(e,t,n){var s=ge,a=pt.transition;try{pt.transition=null,ge=1,ym(e,t,n,s)}finally{pt.transition=a,ge=s}return null}function ym(e,t,n,s){do tn();while(rr!==null);if(fe&6)throw Error(U(327));n=e.finishedWork;var a=e.finishedLanes;if(n===null)return null;if(e.finishedWork=null,e.finishedLanes=0,n===e.current)throw Error(U(177));e.callbackNode=null,e.callbackPriority=0;var l=n.lanes|n.childLanes;if(Jp(e,l),e===Me&&(ze=Me=null,Fe=0),!(n.subtreeFlags&2064)&&!(n.flags&2064)||Is||(Is=!0,Du(Ks,function(){return tn(),null})),l=(n.flags&15990)!==0,n.subtreeFlags&15990||l){l=pt.transition,pt.transition=null;var o=ge;ge=1;var i=fe;fe|=4,Ho.current=null,fm(e,n),Eu(n,e),Af(Fl),Js=!!Ll,Fl=Ll=null,e.current=n,mm(n),Bp(),fe=i,ge=o,pt.transition=l}else e.current=n;if(Is&&(Is=!1,rr=e,ma=a),l=e.pendingLanes,l===0&&(ir=null),Hp(n.stateNode),Je(e,_e()),t!==null)for(s=e.onRecoverableError,n=0;n<t.length;n++)a=t[n],s(a.value,{componentStack:a.stack,digest:a.digest});if(fa)throw fa=!1,e=ro,ro=null,e;return ma&1&&e.tag!==0&&tn(),l=e.pendingLanes,l&1?e===no?Vn++:(Vn=0,no=e):Vn=0,vr(),null}function tn(){if(rr!==null){var e=md(ma),t=pt.transition,n=ge;try{if(pt.transition=null,ge=16>e?16:e,rr===null)var s=!1;else{if(e=rr,rr=null,ma=0,fe&6)throw Error(U(331));var a=fe;for(fe|=4,q=e.current;q!==null;){var l=q,o=l.child;if(q.flags&16){var i=l.deletions;if(i!==null){for(var d=0;d<i.length;d++){var u=i[d];for(q=u;q!==null;){var y=q;switch(y.tag){case 0:case 11:case 15:$n(8,y,l)}var g=y.child;if(g!==null)g.return=y,q=g;else for(;q!==null;){y=q;var x=y.sibling,k=y.return;if(Nu(y),y===u){q=null;break}if(x!==null){x.return=k,q=x;break}q=k}}}var S=l.alternate;if(S!==null){var z=S.child;if(z!==null){S.child=null;do{var R=z.sibling;z.sibling=null,z=R}while(z!==null)}}q=l}}if(l.subtreeFlags&2064&&o!==null)o.return=l,q=o;else e:for(;q!==null;){if(l=q,l.flags&2048)switch(l.tag){case 0:case 11:case 15:$n(9,l,l.return)}var f=l.sibling;if(f!==null){f.return=l.return,q=f;break e}q=l.return}}var p=e.current;for(q=p;q!==null;){o=q;var m=o.child;if(o.subtreeFlags&2064&&m!==null)m.return=o,q=m;else e:for(o=p;q!==null;){if(i=q,i.flags&2048)try{switch(i.tag){case 0:case 11:case 15:_a(9,i)}}catch(j){Ce(i,i.return,j)}if(i===o){q=null;break e}var h=i.sibling;if(h!==null){h.return=i.return,q=h;break e}q=i.return}}if(fe=a,vr(),Tt&&typeof Tt.onPostCommitFiberRoot=="function")try{Tt.onPostCommitFiberRoot(ya,e)}catch{}s=!0}return s}finally{ge=n,pt.transition=t}}return!1}function fc(e,t,n){t=on(n,t),t=mu(e,t,1),e=or(e,t,1),t=Ge(),e!==null&&(cs(e,1,t),Je(e,t))}function Ce(e,t,n){if(e.tag===3)fc(e,e,n);else for(;t!==null;){if(t.tag===3){fc(t,e,n);break}else if(t.tag===1){var s=t.stateNode;if(typeof t.type.getDerivedStateFromError=="function"||typeof s.componentDidCatch=="function"&&(ir===null||!ir.has(s))){e=on(n,e),e=hu(t,e,1),t=or(t,e,1),e=Ge(),t!==null&&(cs(t,1,e),Je(t,e));break}}t=t.return}}function jm(e,t,n){var s=e.pingCache;s!==null&&s.delete(t),t=Ge(),e.pingedLanes|=e.suspendedLanes&n,Me===e&&(Fe&n)===n&&(Pe===4||Pe===3&&(Fe&130023424)===Fe&&500>_e()-Xo?Cr(e,0):Qo|=n),Je(e,t)}function Lu(e,t){t===0&&(e.mode&1?(t=ws,ws<<=1,!(ws&130023424)&&(ws=4194304)):t=1);var n=Ge();e=Wt(e,t),e!==null&&(cs(e,t,n),Je(e,n))}function bm(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),Lu(e,n)}function wm(e,t){var n=0;switch(e.tag){case 13:var s=e.stateNode,a=e.memoizedState;a!==null&&(n=a.retryLane);break;case 19:s=e.stateNode;break;default:throw Error(U(314))}s!==null&&s.delete(t),Lu(e,n)}var Fu;Fu=function(e,t,n){if(e!==null)if(e.memoizedProps!==t.pendingProps||Ke.current)Ye=!0;else{if(!(e.lanes&n)&&!(t.flags&128))return Ye=!1,cm(e,t,n);Ye=!!(e.flags&131072)}else Ye=!1,we&&t.flags&1048576&&$d(t,aa,t.index);switch(t.lanes=0,t.tag){case 2:var s=t.type;Bs(e,t),e=t.pendingProps;var a=nn(t,Ve.current);en(t,n),a=Uo(null,t,s,e,a,n);var l=Vo();return t.flags|=1,typeof a=="object"&&a!==null&&typeof a.render=="function"&&a.$$typeof===void 0?(t.tag=1,t.memoizedState=null,t.updateQueue=null,qe(s)?(l=!0,na(t)):l=!1,t.memoizedState=a.state!==null&&a.state!==void 0?a.state:null,Fo(t),a.updater=Ca,t.stateNode=a,a._reactInternals=t,Gl(t,s,e,n),t=Xl(null,t,s,!0,l,n)):(t.tag=0,we&&l&&zo(t),We(null,t,a,n),t=t.child),t;case 16:s=t.elementType;e:{switch(Bs(e,t),e=t.pendingProps,a=s._init,s=a(s._payload),t.type=s,a=t.tag=Sm(s),e=vt(s,e),a){case 0:t=Ql(null,t,s,e,n);break e;case 1:t=nc(null,t,s,e,n);break e;case 11:t=tc(null,t,s,e,n);break e;case 14:t=rc(null,t,s,vt(s.type,e),n);break e}throw Error(U(306,s,""))}return t;case 0:return s=t.type,a=t.pendingProps,a=t.elementType===s?a:vt(s,a),Ql(e,t,s,a,n);case 1:return s=t.type,a=t.pendingProps,a=t.elementType===s?a:vt(s,a),nc(e,t,s,a,n);case 3:e:{if(yu(t),e===null)throw Error(U(387));s=t.pendingProps,l=t.memoizedState,a=l.element,Hd(e,t),ia(t,s,null,n);var o=t.memoizedState;if(s=o.element,l.isDehydrated)if(l={element:s,isDehydrated:!1,cache:o.cache,pendingSuspenseBoundaries:o.pendingSuspenseBoundaries,transitions:o.transitions},t.updateQueue.baseState=l,t.memoizedState=l,t.flags&256){a=on(Error(U(423)),t),t=sc(e,t,s,n,a);break e}else if(s!==a){a=on(Error(U(424)),t),t=sc(e,t,s,n,a);break e}else for(nt=lr(t.stateNode.containerInfo.firstChild),st=t,we=!0,jt=null,n=Wd(t,null,s,n),t.child=n;n;)n.flags=n.flags&-3|4096,n=n.sibling;else{if(sn(),s===a){t=Gt(e,t,n);break e}We(e,t,s,n)}t=t.child}return t;case 5:return Qd(t),e===null&&Vl(t),s=t.type,a=t.pendingProps,l=e!==null?e.memoizedProps:null,o=a.children,Dl(s,a)?o=null:l!==null&&Dl(s,l)&&(t.flags|=32),vu(e,t),We(e,t,o,n),t.child;case 6:return e===null&&Vl(t),null;case 13:return ju(e,t,n);case 4:return Do(t,t.stateNode.containerInfo),s=t.pendingProps,e===null?t.child=an(t,null,s,n):We(e,t,s,n),t.child;case 11:return s=t.type,a=t.pendingProps,a=t.elementType===s?a:vt(s,a),tc(e,t,s,a,n);case 7:return We(e,t,t.pendingProps,n),t.child;case 8:return We(e,t,t.pendingProps.children,n),t.child;case 12:return We(e,t,t.pendingProps.children,n),t.child;case 10:e:{if(s=t.type._context,a=t.pendingProps,l=t.memoizedProps,o=a.value,ye(la,s._currentValue),s._currentValue=o,l!==null)if(kt(l.value,o)){if(l.children===a.children&&!Ke.current){t=Gt(e,t,n);break e}}else for(l=t.child,l!==null&&(l.return=t);l!==null;){var i=l.dependencies;if(i!==null){o=l.child;for(var d=i.firstContext;d!==null;){if(d.context===s){if(l.tag===1){d=$t(-1,n&-n),d.tag=2;var u=l.updateQueue;if(u!==null){u=u.shared;var y=u.pending;y===null?d.next=d:(d.next=y.next,y.next=d),u.pending=d}}l.lanes|=n,d=l.alternate,d!==null&&(d.lanes|=n),Bl(l.return,n,t),i.lanes|=n;break}d=d.next}}else if(l.tag===10)o=l.type===t.type?null:l.child;else if(l.tag===18){if(o=l.return,o===null)throw Error(U(341));o.lanes|=n,i=o.alternate,i!==null&&(i.lanes|=n),Bl(o,n,t),o=l.sibling}else o=l.child;if(o!==null)o.return=l;else for(o=l;o!==null;){if(o===t){o=null;break}if(l=o.sibling,l!==null){l.return=o.return,o=l;break}o=o.return}l=o}We(e,t,a.children,n),t=t.child}return t;case 9:return a=t.type,s=t.pendingProps.children,en(t,n),a=ft(a),s=s(a),t.flags|=1,We(e,t,s,n),t.child;case 14:return s=t.type,a=vt(s,t.pendingProps),a=vt(s.type,a),rc(e,t,s,a,n);case 15:return xu(e,t,t.type,t.pendingProps,n);case 17:return s=t.type,a=t.pendingProps,a=t.elementType===s?a:vt(s,a),Bs(e,t),t.tag=1,qe(s)?(e=!0,na(t)):e=!1,en(t,n),fu(t,s,a),Gl(t,s,a,n),Xl(null,t,s,!0,e,n);case 19:return bu(e,t,n);case 22:return gu(e,t,n)}throw Error(U(156,t.tag))};function Du(e,t){return dd(e,t)}function km(e,t,n,s){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=s,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function ut(e,t,n,s){return new km(e,t,n,s)}function Jo(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Sm(e){if(typeof e=="function")return Jo(e)?1:0;if(e!=null){if(e=e.$$typeof,e===go)return 11;if(e===vo)return 14}return 2}function dr(e,t){var n=e.alternate;return n===null?(n=ut(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&14680064,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n}function Hs(e,t,n,s,a,l){var o=2;if(s=e,typeof e=="function")Jo(e)&&(o=1);else if(typeof e=="string")o=5;else e:switch(e){case Ar:return _r(n.children,a,l,t);case xo:o=8,a|=8;break;case hl:return e=ut(12,n,t,a|2),e.elementType=hl,e.lanes=l,e;case xl:return e=ut(13,n,t,a),e.elementType=xl,e.lanes=l,e;case gl:return e=ut(19,n,t,a),e.elementType=gl,e.lanes=l,e;case Qc:return za(n,a,l,t);default:if(typeof e=="object"&&e!==null)switch(e.$$typeof){case Gc:o=10;break e;case Hc:o=9;break e;case go:o=11;break e;case vo:o=14;break e;case qt:o=16,s=null;break e}throw Error(U(130,e==null?e:typeof e,""))}return t=ut(o,n,t,a),t.elementType=e,t.type=s,t.lanes=l,t}function _r(e,t,n,s){return e=ut(7,e,s,t),e.lanes=n,e}function za(e,t,n,s){return e=ut(22,e,s,t),e.elementType=Qc,e.lanes=n,e.stateNode={isHidden:!1},e}function il(e,t,n){return e=ut(6,e,null,t),e.lanes=n,e}function cl(e,t,n){return t=ut(4,e.children!==null?e.children:[],e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}function Nm(e,t,n,s,a){this.tag=t,this.containerInfo=e,this.finishedWork=this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.pendingContext=this.context=null,this.callbackPriority=0,this.eventTimes=Ba(0),this.expirationTimes=Ba(-1),this.entangledLanes=this.finishedLanes=this.mutableReadLanes=this.expiredLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=Ba(0),this.identifierPrefix=s,this.onRecoverableError=a,this.mutableSourceEagerHydrationData=null}function Zo(e,t,n,s,a,l,o,i,d){return e=new Nm(e,t,n,i,d),t===1?(t=1,l===!0&&(t|=8)):t=0,l=ut(3,null,null,t),e.current=l,l.stateNode=e,l.memoizedState={element:s,isDehydrated:n,cache:null,transitions:null,pendingSuspenseBoundaries:null},Fo(l),e}function Cm(e,t,n){var s=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:Or,key:s==null?null:""+s,children:e,containerInfo:t,implementation:n}}function Ou(e){if(!e)return pr;e=e._reactInternals;e:{if(Rr(e)!==e||e.tag!==1)throw Error(U(170));var t=e;do{switch(t.tag){case 3:t=t.stateNode.context;break e;case 1:if(qe(t.type)){t=t.stateNode.__reactInternalMemoizedMergedChildContext;break e}}t=t.return}while(t!==null);throw Error(U(171))}if(e.tag===1){var n=e.type;if(qe(n))return Od(e,n,t)}return t}function Au(e,t,n,s,a,l,o,i,d){return e=Zo(n,s,!0,e,a,l,o,i,d),e.context=Ou(null),n=e.current,s=Ge(),a=cr(n),l=$t(s,a),l.callback=t??null,or(n,l,a),e.current.lanes=a,cs(e,a,s),Je(e,s),e}function Ta(e,t,n,s){var a=t.current,l=Ge(),o=cr(a);return n=Ou(n),t.context===null?t.context=n:t.pendingContext=n,t=$t(l,o),t.payload={element:e},s=s===void 0?null:s,s!==null&&(t.callback=s),e=or(a,t,o),e!==null&&(wt(e,a,o,l),$s(e,a,o)),o}function xa(e){if(e=e.current,!e.child)return null;switch(e.child.tag){case 5:return e.child.stateNode;default:return e.child.stateNode}}function mc(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function ei(e,t){mc(e,t),(e=e.alternate)&&mc(e,t)}function _m(){return null}var $u=typeof reportError=="function"?reportError:function(e){console.error(e)};function ti(e){this._internalRoot=e}Pa.prototype.render=ti.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(U(409));Ta(e,t,null,null)};Pa.prototype.unmount=ti.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;Ir(function(){Ta(null,e,null,null)}),t[Bt]=null}};function Pa(e){this._internalRoot=e}Pa.prototype.unstable_scheduleHydration=function(e){if(e){var t=gd();e={blockedOn:null,target:e,priority:t};for(var n=0;n<Zt.length&&t!==0&&t<Zt[n].priority;n++);Zt.splice(n,0,e),n===0&&yd(e)}};function ri(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function Ia(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11&&(e.nodeType!==8||e.nodeValue!==" react-mount-point-unstable "))}function hc(){}function Em(e,t,n,s,a){if(a){if(typeof s=="function"){var l=s;s=function(){var u=xa(o);l.call(u)}}var o=Au(t,s,e,0,null,!1,!1,"",hc);return e._reactRootContainer=o,e[Bt]=o.current,qn(e.nodeType===8?e.parentNode:e),Ir(),o}for(;a=e.lastChild;)e.removeChild(a);if(typeof s=="function"){var i=s;s=function(){var u=xa(d);i.call(u)}}var d=Zo(e,0,!1,null,null,!1,!1,"",hc);return e._reactRootContainer=d,e[Bt]=d.current,qn(e.nodeType===8?e.parentNode:e),Ir(function(){Ta(t,d,n,s)}),d}function Ma(e,t,n,s,a){var l=n._reactRootContainer;if(l){var o=l;if(typeof a=="function"){var i=a;a=function(){var d=xa(o);i.call(d)}}Ta(t,o,e,a)}else o=Em(n,t,e,a,s);return xa(o)}hd=function(e){switch(e.tag){case 3:var t=e.stateNode;if(t.current.memoizedState.isDehydrated){var n=In(t.pendingLanes);n!==0&&(bo(t,n|1),Je(t,_e()),!(fe&6)&&(cn=_e()+500,vr()))}break;case 13:Ir(function(){var s=Wt(e,1);if(s!==null){var a=Ge();wt(s,e,1,a)}}),ei(e,1)}};wo=function(e){if(e.tag===13){var t=Wt(e,134217728);if(t!==null){var n=Ge();wt(t,e,134217728,n)}ei(e,134217728)}};xd=function(e){if(e.tag===13){var t=cr(e),n=Wt(e,t);if(n!==null){var s=Ge();wt(n,e,t,s)}ei(e,t)}};gd=function(){return ge};vd=function(e,t){var n=ge;try{return ge=e,t()}finally{ge=n}};_l=function(e,t,n){switch(t){case"input":if(jl(e,n),t=n.name,n.type==="radio"&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll("input[name="+JSON.stringify(""+t)+'][type="radio"]'),t=0;t<n.length;t++){var s=n[t];if(s!==e&&s.form===e.form){var a=ka(s);if(!a)throw Error(U(90));Yc(s),jl(s,a)}}}break;case"textarea":qc(e,n);break;case"select":t=n.value,t!=null&&Kr(e,!!n.multiple,t,!1)}};sd=Yo;ad=Ir;var zm={usingClientEntryPoint:!1,Events:[us,Br,ka,rd,nd,Yo]},_n={findFiberByHostInstance:kr,bundleType:0,version:"18.3.1",rendererPackageName:"react-dom"},Tm={bundleType:_n.bundleType,version:_n.version,rendererPackageName:_n.rendererPackageName,rendererConfig:_n.rendererConfig,overrideHookState:null,overrideHookStateDeletePath:null,overrideHookStateRenamePath:null,overrideProps:null,overridePropsDeletePath:null,overridePropsRenamePath:null,setErrorHandler:null,setSuspenseHandler:null,scheduleUpdate:null,currentDispatcherRef:Qt.ReactCurrentDispatcher,findHostInstanceByFiber:function(e){return e=id(e),e===null?null:e.stateNode},findFiberByHostInstance:_n.findFiberByHostInstance||_m,findHostInstancesForRefresh:null,scheduleRefresh:null,scheduleRoot:null,setRefreshHandler:null,getCurrentFiber:null,reconcilerVersion:"18.3.1-next-f1338f8080-20240426"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"){var Ms=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Ms.isDisabled&&Ms.supportsFiber)try{ya=Ms.inject(Tm),Tt=Ms}catch{}}lt.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=zm;lt.createPortal=function(e,t){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!ri(t))throw Error(U(200));return Cm(e,t,null,n)};lt.createRoot=function(e,t){if(!ri(e))throw Error(U(299));var n=!1,s="",a=$u;return t!=null&&(t.unstable_strictMode===!0&&(n=!0),t.identifierPrefix!==void 0&&(s=t.identifierPrefix),t.onRecoverableError!==void 0&&(a=t.onRecoverableError)),t=Zo(e,1,!1,null,null,n,!1,s,a),e[Bt]=t.current,qn(e.nodeType===8?e.parentNode:e),new ti(t)};lt.findDOMNode=function(e){if(e==null)return null;if(e.nodeType===1)return e;var t=e._reactInternals;if(t===void 0)throw typeof e.render=="function"?Error(U(188)):(e=Object.keys(e).join(","),Error(U(268,e)));return e=id(t),e=e===null?null:e.stateNode,e};lt.flushSync=function(e){return Ir(e)};lt.hydrate=function(e,t,n){if(!Ia(t))throw Error(U(200));return Ma(null,e,t,!0,n)};lt.hydrateRoot=function(e,t,n){if(!ri(e))throw Error(U(405));var s=n!=null&&n.hydratedSources||null,a=!1,l="",o=$u;if(n!=null&&(n.unstable_strictMode===!0&&(a=!0),n.identifierPrefix!==void 0&&(l=n.identifierPrefix),n.onRecoverableError!==void 0&&(o=n.onRecoverableError)),t=Au(t,null,e,1,n??null,a,!1,l,o),e[Bt]=t.current,qn(e),s)for(e=0;e<s.length;e++)n=s[e],a=n._getVersion,a=a(n._source),t.mutableSourceEagerHydrationData==null?t.mutableSourceEagerHydrationData=[n,a]:t.mutableSourceEagerHydrationData.push(n,a);return new Pa(t)};lt.render=function(e,t,n){if(!Ia(t))throw Error(U(200));return Ma(null,e,t,!1,n)};lt.unmountComponentAtNode=function(e){if(!Ia(e))throw Error(U(40));return e._reactRootContainer?(Ir(function(){Ma(null,null,e,!1,function(){e._reactRootContainer=null,e[Bt]=null})}),!0):!1};lt.unstable_batchedUpdates=Yo;lt.unstable_renderSubtreeIntoContainer=function(e,t,n,s){if(!Ia(n))throw Error(U(200));if(e==null||e._reactInternals===void 0)throw Error(U(38));return Ma(e,t,n,!1,s)};lt.version="18.3.1-next-f1338f8080-20240426";function Uu(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(Uu)}catch(e){console.error(e)}}Uu(),Uc.exports=lt;var Pm=Uc.exports,xc=Pm;fl.createRoot=xc.createRoot,fl.hydrateRoot=xc.hydrateRoot;/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Im=e=>e.replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase(),Vu=(...e)=>e.filter((t,n,s)=>!!t&&t.trim()!==""&&s.indexOf(t)===n).join(" ").trim();/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */var Mm={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Rm=c.forwardRef(({color:e="currentColor",size:t=24,strokeWidth:n=2,absoluteStrokeWidth:s,className:a="",children:l,iconNode:o,...i},d)=>c.createElement("svg",{ref:d,...Mm,width:t,height:t,stroke:e,strokeWidth:s?Number(n)*24/Number(t):n,className:Vu("lucide",a),...i},[...o.map(([u,y])=>c.createElement(u,y)),...Array.isArray(l)?l:[l]]));/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const se=(e,t)=>{const n=c.forwardRef(({className:s,...a},l)=>c.createElement(Rm,{ref:l,iconNode:t,className:Vu(`lucide-${Im(e)}`,s),...a}));return n.displayName=`${e}`,n};/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Lm=[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"m12 5 7 7-7 7",key:"xquz4c"}]],Fm=se("ArrowRight",Lm);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Dm=[["path",{d:"m21 16-4 4-4-4",key:"f6ql7i"}],["path",{d:"M17 20V4",key:"1ejh1v"}],["path",{d:"m3 8 4-4 4 4",key:"11wl7u"}],["path",{d:"M7 4v16",key:"1glfcx"}]],Om=se("ArrowUpDown",Dm);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Am=[["path",{d:"M20 6 9 17l-5-5",key:"1gmf2c"}]],Bu=se("Check",Am);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const $m=[["path",{d:"m6 9 6 6 6-6",key:"qrunsl"}]],Mt=se("ChevronDown",$m);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Um=[["path",{d:"m15 18-6-6 6-6",key:"1wnfg3"}]],Wu=se("ChevronLeft",Um);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Vm=[["path",{d:"m9 18 6-6-6-6",key:"mthhwq"}]],Gu=se("ChevronRight",Vm);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Bm=[["path",{d:"m18 15-6-6-6 6",key:"153udz"}]],Wm=se("ChevronUp",Bm);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Gm=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["line",{x1:"12",x2:"12",y1:"8",y2:"12",key:"1pkeuh"}],["line",{x1:"12",x2:"12.01",y1:"16",y2:"16",key:"4dfq90"}]],Hm=se("CircleAlert",Gm);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Qm=[["path",{d:"M21.801 10A10 10 0 1 1 17 3.335",key:"yps3ct"}],["path",{d:"m9 11 3 3L22 4",key:"1pflzl"}]],Xm=se("CircleCheckBig",Qm);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ym=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"m9 12 2 2 4-4",key:"dzmm74"}]],Km=se("CircleCheck",Ym);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const qm=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3",key:"1u773s"}],["path",{d:"M12 17h.01",key:"p32p05"}]],Hu=se("CircleHelp",qm);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Jm=[["path",{d:"M20.2 6 3 11l-.9-2.4c-.3-1.1.3-2.2 1.3-2.5l13.5-4c1.1-.3 2.2.3 2.5 1.3Z",key:"1tn4o7"}],["path",{d:"m6.2 5.3 3.1 3.9",key:"iuk76l"}],["path",{d:"m12.4 3.4 3.1 4",key:"6hsd6n"}],["path",{d:"M3 11h18v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z",key:"ltgou9"}]],Zm=se("Clapperboard",Jm);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const eh=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["polyline",{points:"12 6 12 12 16 14",key:"68esgv"}]],Ra=se("Clock",eh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const th=[["rect",{width:"14",height:"14",x:"8",y:"8",rx:"2",ry:"2",key:"17jyea"}],["path",{d:"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2",key:"zix9uf"}]],At=se("Copy",th);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const rh=[["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",key:"ih7n3h"}],["polyline",{points:"7 10 12 15 17 10",key:"2ggqvy"}],["line",{x1:"12",x2:"12",y1:"15",y2:"3",key:"1vk2je"}]],fr=se("Download",rh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const nh=[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]],gc=se("ExternalLink",nh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const sh=[["path",{d:"M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",key:"1rqfz7"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"M10 12a1 1 0 0 0-1 1v1a1 1 0 0 1-1 1 1 1 0 0 1 1 1v1a1 1 0 0 0 1 1",key:"1oajmo"}],["path",{d:"M14 18a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1 1 1 0 0 1-1-1v-1a1 1 0 0 0-1-1",key:"mpwhp6"}]],vc=se("FileJson",sh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ah=[["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"M4.268 21a2 2 0 0 0 1.727 1H18a2 2 0 0 0 2-2V7l-5-5H6a2 2 0 0 0-2 2v3",key:"ms7g94"}],["path",{d:"m9 18-1.5-1.5",key:"1j6qii"}],["circle",{cx:"5",cy:"14",r:"3",key:"ufru5t"}]],lh=se("FileSearch",ah);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const oh=[["path",{d:"M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",key:"1rqfz7"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"M10 9H8",key:"b1mrlr"}],["path",{d:"M16 13H8",key:"t4e002"}],["path",{d:"M16 17H8",key:"z1uh3a"}]],yc=se("FileText",oh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ih=[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",key:"afitv7"}],["path",{d:"M7 3v18",key:"bbkbws"}],["path",{d:"M3 7.5h4",key:"zfgn84"}],["path",{d:"M3 12h18",key:"1i2n21"}],["path",{d:"M3 16.5h4",key:"1230mu"}],["path",{d:"M17 3v18",key:"in4fa5"}],["path",{d:"M17 7.5h4",key:"myr1c1"}],["path",{d:"M17 16.5h4",key:"go4c1d"}]],ls=se("Film",ih);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ch=[["polygon",{points:"22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3",key:"1yg77f"}]],dh=se("Filter",ch);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const uh=[["path",{d:"m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2",key:"usdka0"}]],Qu=se("FolderOpen",uh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ph=[["line",{x1:"22",x2:"2",y1:"6",y2:"6",key:"15w7dq"}],["line",{x1:"22",x2:"2",y1:"18",y2:"18",key:"1ip48p"}],["line",{x1:"6",x2:"6",y1:"2",y2:"22",key:"a2lnyx"}],["line",{x1:"18",x2:"18",y1:"2",y2:"22",key:"8vb6jd"}]],fh=se("Frame",ph);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const mh=[["path",{d:"M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z",key:"c3ymky"}]],dl=se("Heart",mh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const hh=[["path",{d:"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8",key:"1357e3"}],["path",{d:"M3 3v5h5",key:"1xhq8a"}],["path",{d:"M12 7v5l4 2",key:"1fdv2h"}]],xh=se("History",hh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const gh=[["path",{d:"M16 5h6",key:"1vod17"}],["path",{d:"M19 2v6",key:"4bpg5p"}],["path",{d:"M21 11.5V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7.5",key:"1ue2ih"}],["path",{d:"m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21",key:"1xmnt7"}],["circle",{cx:"9",cy:"9",r:"2",key:"af1f0g"}]],vh=se("ImagePlus",gh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const yh=[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",ry:"2",key:"1m3agn"}],["circle",{cx:"9",cy:"9",r:"2",key:"af1f0g"}],["path",{d:"m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21",key:"1xmnt7"}]],mr=se("Image",yh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const jh=[["path",{d:"M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z",key:"zw3jo"}],["path",{d:"M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12",key:"1wduqc"}],["path",{d:"M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17",key:"kqbvx6"}]],ni=se("Layers",jh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const bh=[["path",{d:"M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71",key:"1cjeqo"}],["path",{d:"M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",key:"19qd67"}]],wh=se("Link",bh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const kh=[["path",{d:"M21 12a9 9 0 1 1-6.219-8.56",key:"13zald"}]],Oe=se("LoaderCircle",kh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Sh=[["polyline",{points:"15 3 21 3 21 9",key:"mznyad"}],["polyline",{points:"9 21 3 21 3 15",key:"1avn1i"}],["line",{x1:"21",x2:"14",y1:"3",y2:"10",key:"ota7mn"}],["line",{x1:"3",x2:"10",y1:"21",y2:"14",key:"1atl0r"}]],Xu=se("Maximize2",Sh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Nh=[["path",{d:"M7.9 20A9 9 0 1 0 4 16.1L2 22Z",key:"vv11sd"}]],Ch=se("MessageCircle",Nh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const _h=[["path",{d:"M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z",key:"131961"}],["path",{d:"M19 10v2a7 7 0 0 1-14 0v-2",key:"1vc78b"}],["line",{x1:"12",x2:"12",y1:"19",y2:"22",key:"x3vr5v"}]],Eh=se("Mic",_h);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const zh=[["polyline",{points:"4 14 10 14 10 20",key:"11kfnr"}],["polyline",{points:"20 10 14 10 14 4",key:"rlmsce"}],["line",{x1:"14",x2:"21",y1:"10",y2:"3",key:"o5lafz"}],["line",{x1:"3",x2:"10",y1:"21",y2:"14",key:"1atl0r"}]],Th=se("Minimize2",zh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ph=[["path",{d:"M12 2v20",key:"t6zp3m"}],["path",{d:"m15 19-3 3-3-3",key:"11eu04"}],["path",{d:"m19 9 3 3-3 3",key:"1mg7y2"}],["path",{d:"M2 12h20",key:"9i4pu4"}],["path",{d:"m5 9-3 3 3 3",key:"j64kie"}],["path",{d:"m9 5 3-3 3 3",key:"l8vdw6"}]],Ih=se("Move",Ph);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Mh=[["path",{d:"M9 18V5l12-2v13",key:"1jmyc2"}],["circle",{cx:"6",cy:"18",r:"3",key:"fqmcym"}],["circle",{cx:"18",cy:"16",r:"3",key:"1hluhg"}]],Rh=se("Music",Mh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Lh=[["rect",{x:"14",y:"4",width:"4",height:"16",rx:"1",key:"zuxfzm"}],["rect",{x:"6",y:"4",width:"4",height:"16",rx:"1",key:"1okwgv"}]],Fh=se("Pause",Lh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Dh=[["polygon",{points:"6 3 20 12 6 21 6 3",key:"1oa8hb"}]],si=se("Play",Dh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Oh=[["path",{d:"M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8",key:"v9h5vc"}],["path",{d:"M21 3v5h-5",key:"1q7to0"}],["path",{d:"M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16",key:"3uifl3"}],["path",{d:"M8 16H3v5",key:"1cv678"}]],dn=se("RefreshCw",Oh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Ah=[["path",{d:"M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z",key:"1ffxy3"}],["path",{d:"m21.854 2.147-10.94 10.939",key:"12cjpa"}]],Yu=se("Send",Ah);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const $h=[["path",{d:"M20 7h-9",key:"3s1dr2"}],["path",{d:"M14 17H5",key:"gfn3mx"}],["circle",{cx:"17",cy:"17",r:"3",key:"18b49y"}],["circle",{cx:"7",cy:"7",r:"3",key:"dfmy0x"}]],Uh=se("Settings2",$h);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Vh=[["path",{d:"M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z",key:"1qme2f"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]],hr=se("Settings",Vh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Bh=[["line",{x1:"4",x2:"4",y1:"21",y2:"14",key:"1p332r"}],["line",{x1:"4",x2:"4",y1:"10",y2:"3",key:"gb41h5"}],["line",{x1:"12",x2:"12",y1:"21",y2:"12",key:"hf2csr"}],["line",{x1:"12",x2:"12",y1:"8",y2:"3",key:"1kfi7u"}],["line",{x1:"20",x2:"20",y1:"21",y2:"16",key:"1lhrwl"}],["line",{x1:"20",x2:"20",y1:"12",y2:"3",key:"16vvfq"}],["line",{x1:"2",x2:"6",y1:"14",y2:"14",key:"1uebub"}],["line",{x1:"10",x2:"14",y1:"8",y2:"8",key:"1yglbp"}],["line",{x1:"18",x2:"22",y1:"16",y2:"16",key:"1jxqpz"}]],ga=se("SlidersVertical",Bh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Wh=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M8 14s1.5 2 4 2 4-2 4-2",key:"1y1vjs"}],["line",{x1:"9",x2:"9.01",y1:"9",y2:"9",key:"yxxnd0"}],["line",{x1:"15",x2:"15.01",y1:"9",y2:"9",key:"1p4y9e"}]],Gh=se("Smile",Wh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Hh=[["path",{d:"M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z",key:"4pj2yx"}],["path",{d:"M20 3v4",key:"1olli1"}],["path",{d:"M22 5h-4",key:"1gvqau"}],["path",{d:"M4 17v2",key:"vumght"}],["path",{d:"M5 18H3",key:"zchphs"}]],Ut=se("Sparkles",Hh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Qh=[["polyline",{points:"4 17 10 11 4 5",key:"akl6gq"}],["line",{x1:"12",x2:"20",y1:"19",y2:"19",key:"q2wloq"}]],jc=se("Terminal",Qh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Xh=[["path",{d:"M3 6h18",key:"d0wm0j"}],["path",{d:"M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6",key:"4alrt4"}],["path",{d:"M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2",key:"v07s0e"}],["line",{x1:"10",x2:"10",y1:"11",y2:"17",key:"1uufr5"}],["line",{x1:"14",x2:"14",y1:"11",y2:"17",key:"xtxkd"}]],Yh=se("Trash2",Xh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Kh=[["polyline",{points:"4 7 4 4 20 4 20 7",key:"1nosan"}],["line",{x1:"9",x2:"15",y1:"20",y2:"20",key:"swin9y"}],["line",{x1:"12",x2:"12",y1:"4",y2:"20",key:"1tx1rr"}]],Ku=se("Type",Kh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const qh=[["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",key:"ih7n3h"}],["polyline",{points:"17 8 12 3 7 8",key:"t8dd8p"}],["line",{x1:"12",x2:"12",y1:"3",y2:"15",key:"widbto"}]],St=se("Upload",qh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Jh=[["path",{d:"M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2",key:"975kel"}],["circle",{cx:"12",cy:"7",r:"4",key:"17ys0d"}]],lo=se("User",Jh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const Zh=[["path",{d:"m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5",key:"ftymec"}],["rect",{x:"2",y:"6",width:"14",height:"12",rx:"2",key:"158x01"}]],os=se("Video",Zh);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ex=[["path",{d:"M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z",key:"uqj9uw"}],["path",{d:"M16 9a5 5 0 0 1 0 6",key:"1q6k2b"}],["path",{d:"M19.364 18.364a9 9 0 0 0 0-12.728",key:"ijwkga"}]],oo=se("Volume2",ex);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const tx=[["path",{d:"m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72",key:"ul74o6"}],["path",{d:"m14 7 3 3",key:"1r5n42"}],["path",{d:"M5 6v4",key:"ilb8ba"}],["path",{d:"M19 14v4",key:"blhpug"}],["path",{d:"M10 2v2",key:"7u0qdc"}],["path",{d:"M7 8H3",key:"zfb6yr"}],["path",{d:"M21 16h-4",key:"1cnmox"}],["path",{d:"M11 3H9",key:"1obp7u"}]],Ht=se("WandSparkles",tx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const rx=[["path",{d:"M12 20h.01",key:"zekei9"}],["path",{d:"M8.5 16.429a5 5 0 0 1 7 0",key:"1bycff"}],["path",{d:"M5 12.859a10 10 0 0 1 5.17-2.69",key:"1dl1wf"}],["path",{d:"M19 12.859a10 10 0 0 0-2.007-1.523",key:"4k23kn"}],["path",{d:"M2 8.82a15 15 0 0 1 4.177-2.643",key:"1grhjp"}],["path",{d:"M22 8.82a15 15 0 0 0-11.288-3.764",key:"z3jwby"}],["path",{d:"m2 2 20 20",key:"1ooewy"}]],nx=se("WifiOff",rx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const sx=[["path",{d:"M12 20h.01",key:"zekei9"}],["path",{d:"M2 8.82a15 15 0 0 1 20 0",key:"dnpr2z"}],["path",{d:"M5 12.859a10 10 0 0 1 14 0",key:"1x1e6c"}],["path",{d:"M8.5 16.429a5 5 0 0 1 7 0",key:"1bycff"}]],ax=se("Wifi",sx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const lx=[["rect",{width:"8",height:"8",x:"3",y:"3",rx:"2",key:"by2w9f"}],["path",{d:"M7 11v4a2 2 0 0 0 2 2h4",key:"xkn7yn"}],["rect",{width:"8",height:"8",x:"13",y:"13",rx:"2",key:"1cgmvn"}]],qu=se("Workflow",lx);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ox=[["path",{d:"M18 6 6 18",key:"1bl5f8"}],["path",{d:"m6 6 12 12",key:"d8bk6v"}]],It=se("X",ox);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const ix=[["path",{d:"M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z",key:"1xq2db"}]],Ju=se("Zap",ix);/**
 * @license lucide-react v0.474.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const cx=[["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}],["line",{x1:"21",x2:"16.65",y1:"21",y2:"16.65",key:"13gj7c"}],["line",{x1:"11",x2:"11",y1:"8",y2:"14",key:"1vmskp"}],["line",{x1:"8",x2:"14",y1:"11",y2:"11",key:"durymu"}]],bc=se("ZoomIn",cx),ee="http://192.168.1.2:7998",Ee=!1,ne={TEXT_TO_IMAGE:"text-to-image",TEXT_TO_VIDEO:"text-to-video",IMAGE_TO_VIDEO:"image-to-video",TEXT_TO_IMAGE_TO_VIDEO:"text-to-image-to-video",VIDEO_TO_VIDEO:"video-to-video",VIDEO_TO_TEXT:"video-to-text",IMAGE_TO_IMAGE:"image-to-image",REFRAME:"reframe",FACE_SWAP:"face-swap",UPSCALER:"upscaler",IMAGE_TO_TEXT:"image-to-text",PROMPT_GENERATOR:"prompt-generator",AUDIO_GENERATION:"audio-generation",PIPELINE:"pipeline",LORA_TRAINING:"lora-training",MY_MEDIA_ALL:"my-media-all",MY_MEDIA_VIDEOS:"my-media-videos",MY_MEDIA_IMAGES:"my-media-images",MY_MEDIA_PROMPTS:"my-media-prompts"},dx=[{id:"video-tools",title:"Video Tools",items:[{id:ne.IMAGE_TO_VIDEO,label:"Image to Video",status:"ready"},{id:ne.TEXT_TO_VIDEO,label:"Text to Video",status:"ready"},{id:ne.TEXT_TO_IMAGE_TO_VIDEO,label:"Text to Image to Video",status:"ready"},{id:ne.VIDEO_TO_VIDEO,label:"Video to Video",status:"ready"},{id:ne.VIDEO_TO_TEXT,label:"Video to Text",status:"new"}]},{id:"image-tools",title:"Image Tools",items:[{id:ne.TEXT_TO_IMAGE,label:"Text to Image",status:"ready"},{id:ne.IMAGE_TO_IMAGE,label:"Image to Image",status:"ready"},{id:ne.UPSCALER,label:"Upscaler",status:"ready"},{id:ne.REFRAME,label:"Reframe",status:"new"},{id:ne.FACE_SWAP,label:"Face Swap",status:"new"}]},{id:"prompt-tools",title:"Prompt Tools",items:[{id:ne.IMAGE_TO_TEXT,label:"Image to Text",status:"new"},{id:ne.PROMPT_GENERATOR,label:"Prompt Generator",status:"new"}]},{id:"audio-tools",title:"Audio Tools",items:[{id:ne.AUDIO_GENERATION,label:"Audio Generation",status:"new"}]},{id:"advanced",title:"Advanced",items:[{id:ne.PIPELINE,label:"Pipeline",status:"ready"},{id:ne.LORA_TRAINING,label:"LoRA Training",status:"ready"}]},{id:"my-media",title:"My Media",items:[{id:ne.MY_MEDIA_ALL,label:"All",status:"ready"},{id:ne.MY_MEDIA_VIDEOS,label:"Videos",status:"ready"},{id:ne.MY_MEDIA_IMAGES,label:"Images",status:"ready"},{id:ne.MY_MEDIA_PROMPTS,label:"Prompts",status:"ready"}]}],ux={"text-to-video":os,"image-to-video":ls,"text-to-image-to-video":Zm,pipeline:qu,"video-to-video":ni,"text-to-image":Ku,"image-to-image":mr,reframe:Xu,"face-swap":lo,upscaler:Ht,"lora-training":dn,"my-media-all":Qu,"my-media-videos":si,"my-media-images":vh};function px({activeToolId:e,onSelectTool:t,collapsed:n,onToggleCollapsed:s}){return r.jsxs("aside",{className:`sidebar ${n?"collapsed":""}`,children:[r.jsx("div",{className:"sidebar-header",children:r.jsx("div",{className:"sidebar-logo",children:"Oelala"})}),r.jsx("nav",{className:"sidebar-nav",children:dx.map(a=>r.jsxs("div",{className:"sidebar-group",children:[r.jsx("div",{className:"sidebar-group-title",children:a.title}),a.items.map(l=>{const o=e===l.id,i=ux[l.id]||Ht;return r.jsxs("button",{className:`nav-item${o?" active":""}`,onClick:()=>t(l.id),type:"button",children:[r.jsx("span",{className:"nav-icon",children:r.jsx(i,{size:18})}),r.jsx("span",{className:"nav-label",children:l.label}),l.status==="missing-backend"&&r.jsx("span",{className:"nav-badge",children:"v2"})]},l.id)})]},a.id))}),r.jsx("div",{className:"sidebar-footer",children:r.jsxs("button",{onClick:s,className:"nav-item collapse-btn",children:[r.jsx("span",{className:"nav-icon",children:n?r.jsx(Gu,{size:18}):r.jsx(Wu,{size:18})}),r.jsx("span",{className:"nav-label",children:"Collapse"})]})})]})}async function La(e){try{await fetch(`${ee}/client-log`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(e)})}catch(t){console.error("Failed to send client log",t)}}function fx(e){const[t,n]=c.useState([]),[s,a]=c.useState(!1),[l,o]=c.useState(""),i=c.useCallback(async()=>{a(!0),o("");try{const d=await fetch(`${ee}/list-videos`),u=await d.json();if(!d.ok)throw new Error((u==null?void 0:u.detail)||`History failed (${d.status})`);n(Array.isArray(u==null?void 0:u.videos)?u.videos:[])}catch(d){const u=(d==null?void 0:d.message)||"Failed to load history";o(u),await La({level:"error",message:"History fetch failed",timestamp:new Date().toISOString(),meta:{message:u}})}finally{a(!1)}},[]);return c.useEffect(()=>{i()},[i,e]),{videos:t,loading:s,error:l,refresh:i}}function wc(e){const t=Math.floor(e/60),n=Math.floor(e%60);return`${t}:${n.toString().padStart(2,"0")}`}function mx({output:e,refreshToken:t,onSelectHistoryVideo:n,onClose:s}){const[a,l]=c.useState(!1),[o,i]=c.useState(null),[d,u]=c.useState(!1),{videos:y,loading:g,error:x}=fx(t),k=c.useMemo(()=>e?e.kind==="video"?r.jsxs("div",{className:"media-container",children:[r.jsxs("div",{className:"video-wrapper",onMouseEnter:()=>u(!0),onMouseLeave:()=>u(!1),children:[r.jsx("video",{className:"media-preview",controls:!0,src:e.url,autoPlay:!0,loop:!0,onLoadedMetadata:S=>i(S.target.duration)}),d&&o&&r.jsxs("div",{className:"video-duration-overlay",children:[r.jsx(Ra,{size:14}),r.jsx("span",{children:wc(o)})]})]}),r.jsxs("div",{className:"media-info",children:[r.jsxs("div",{className:"media-meta",children:[e.filename||"Generated Video",o&&r.jsxs("span",{className:"duration-inline",children:[" • ",wc(o)]})]}),r.jsxs("div",{className:"media-actions",children:[e.url&&r.jsx("a",{className:"icon-btn",href:e.url,download:e.filename||void 0,title:"Download",children:r.jsx(fr,{size:18})}),e.backendUrl&&r.jsx("a",{className:"icon-btn",href:e.backendUrl,target:"_blank",rel:"noreferrer",title:"Open in new tab",children:r.jsx(gc,{size:18})})]})]})]}):e.kind==="image"?r.jsxs("div",{className:"media-container",children:[r.jsx("img",{className:"media-preview",src:e.url,alt:"Generated",onError:S=>{console.error("Image load failed:",e.url),S.target.style.display="none",S.target.parentNode.innerHTML+=`<div style="padding:20px;color:red">Failed to load image: ${e.url}</div>`}}),r.jsxs("div",{className:"media-info",children:[r.jsx("div",{className:"media-meta",children:e.filename||"Generated Image"}),r.jsxs("div",{className:"media-actions",children:[e.url&&r.jsx("a",{className:"icon-btn",href:e.url,download:e.filename||void 0,title:"Download",children:r.jsx(fr,{size:18})}),e.backendUrl&&r.jsx("a",{className:"icon-btn",href:e.backendUrl,target:"_blank",rel:"noreferrer",title:"Open in new tab",children:r.jsx(gc,{size:18})})]})]})]}):e.kind==="lora"?r.jsxs("div",{className:"media-container",style:{padding:"24px"},children:[r.jsx("h3",{children:"LoRA Training Complete"}),r.jsxs("div",{className:"media-meta",style:{marginTop:"16px"},children:[r.jsxs("p",{children:["ID: ",e.training_id]}),r.jsxs("p",{children:["Path: ",e.lora_path]})]})]}):null:r.jsxs("div",{className:"placeholder-state",children:[r.jsx("div",{className:"placeholder-icon",children:r.jsx(ls,{})}),r.jsx("h3",{children:"Ready to Create"}),r.jsx("p",{className:"muted",children:"Configure parameters and click Generate"})]}),[e]);return r.jsxs("section",{className:"output-panel",children:[r.jsxs("div",{style:{position:"absolute",top:20,right:20,zIndex:10,display:"flex",gap:"8px"},children:[r.jsx("button",{className:"icon-btn",onClick:()=>l(!a),title:"History",children:r.jsx(xh,{size:20})}),s&&r.jsx("button",{className:"icon-btn",onClick:s,title:"Close & show My Media",children:r.jsx(It,{size:20})})]}),k,a&&r.jsxs("div",{className:"history",children:[r.jsxs("div",{className:"history-title",style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsx("span",{children:"History"}),r.jsx("button",{className:"icon-btn",onClick:()=>l(!1),children:r.jsx(It,{size:18})})]}),r.jsxs("div",{className:"history-list",children:[g&&r.jsx("div",{style:{padding:20,textAlign:"center"},className:"muted",children:"Loading..."}),x&&r.jsx("div",{className:"error",children:x}),!g&&!x&&y.length===0&&r.jsx("div",{style:{padding:20,textAlign:"center"},className:"muted",children:"No history yet"}),y.map(S=>r.jsxs("button",{className:"history-item",onClick:()=>{n({kind:"video",url:`${ee}/outputs/${S.filename}`,backendUrl:`${ee}/outputs/${S.filename}`,filename:S.filename})},children:[r.jsx("div",{className:"history-item-title",children:S.filename}),r.jsx("div",{className:"history-item-sub",children:new Date(S.mtime*1e3).toLocaleString()})]},S.filename))]})]})]})}function hx({onJobComplete:e,refreshToken:t}){const[n,s]=c.useState({running:[],pending:[],total_running:0,total_pending:0}),[a,l]=c.useState([]),[o,i]=c.useState(!1),[d,u]=c.useState(new Set),y=c.useRef(null),g=c.useCallback(async()=>{try{const R=await fetch(`${ee}/comfyui/queue`);if(!R.ok)return;const f=await R.json();s(f)}catch{}},[]),x=c.useCallback(async R=>{try{const f=await fetch(`${ee}/comfyui/job/${R}`);return f.ok?await f.json():null}catch{return null}},[]);c.useEffect(()=>{g();const R=setInterval(g,3e3);return()=>clearInterval(R)},[g,t]),c.useEffect(()=>{for(const R of a)!d.has(R.prompt_id)&&R.status==="completed"&&R.output_video&&(e&&e(R),u(f=>new Set([...f,R.prompt_id])))},[a,d,e]),c.useEffect(()=>{const R=async()=>{for(const f of n.running){const p=await x(f.prompt_id);p&&p.status==="completed"&&l(m=>m.some(h=>h.prompt_id===p.prompt_id)?m:[...m,p].slice(-10))}};n.running.length>0&&R()},[n.running,x]),c.useEffect(()=>{const R=f=>{y.current&&!y.current.contains(f.target)&&i(!1)};if(o)return document.addEventListener("mousedown",R),()=>document.removeEventListener("mousedown",R)},[o]);const k=async R=>{try{await fetch(`${ee}/comfyui/queue/${R}`,{method:"DELETE"}),g()}catch(f){console.error("Failed to cancel job:",f)}},S=n.total_running>0,z=n.total_running+n.total_pending;return r.jsxs("div",{style:{position:"relative"},ref:y,children:[r.jsxs("button",{onClick:()=>i(!o),style:{display:"flex",alignItems:"center",gap:"6px",padding:"6px 10px",backgroundColor:S?"rgba(34, 197, 94, 0.15)":"transparent",border:`1px solid ${S?"#22c55e":"var(--border-color)"}`,borderRadius:"6px",cursor:"pointer",color:"var(--text-primary)",fontSize:"0.8rem"},title:S?`${n.total_running} running, ${n.total_pending} queued`:"No active jobs",children:[S?r.jsx(Oe,{size:14,color:"#22c55e",className:"spin"}):r.jsx(Ra,{size:14,color:"var(--text-muted)"}),r.jsx("span",{style:{fontWeight:500},children:S?n.total_running:0}),n.total_pending>0&&r.jsxs("span",{style:{color:"var(--text-muted)"},children:["+",n.total_pending]})]}),o&&r.jsxs("div",{style:{position:"absolute",top:"100%",right:0,marginTop:"8px",width:"320px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"8px",boxShadow:"0 4px 20px rgba(0,0,0,0.3)",zIndex:1e3,overflow:"hidden"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"10px 12px",borderBottom:"1px solid var(--border-color)",backgroundColor:"var(--bg-primary)"},children:[r.jsx("span",{style:{fontWeight:600,fontSize:"0.85rem"},children:"Generation Queue"}),r.jsxs("div",{style:{display:"flex",gap:"8px"},children:[r.jsx("button",{onClick:g,style:{background:"transparent",border:"none",cursor:"pointer",padding:"2px"},children:r.jsx(dn,{size:12,color:"var(--text-muted)"})}),r.jsx("button",{onClick:()=>i(!1),style:{background:"transparent",border:"none",cursor:"pointer",padding:"2px"},children:r.jsx(It,{size:14,color:"var(--text-muted)"})})]})]}),r.jsxs("div",{style:{maxHeight:"300px",overflowY:"auto",padding:"8px"},children:[n.running.length>0&&r.jsxs("div",{style:{marginBottom:"8px"},children:[r.jsx("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginBottom:"4px",textTransform:"uppercase"},children:"Running"}),n.running.map(R=>r.jsx(ul,{job:R,status:"running",onCancel:k},R.prompt_id))]}),n.pending.length>0&&r.jsxs("div",{style:{marginBottom:"8px"},children:[r.jsx("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginBottom:"4px",textTransform:"uppercase"},children:"Pending"}),n.pending.map(R=>r.jsx(ul,{job:R,status:"pending",onCancel:k},R.prompt_id))]}),a.length>0&&r.jsxs("div",{children:[r.jsx("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginBottom:"4px",textTransform:"uppercase"},children:"Completed"}),a.slice(-3).reverse().map(R=>r.jsx(ul,{job:R,status:"completed"},R.prompt_id))]}),z===0&&a.length===0&&r.jsx("div",{style:{textAlign:"center",padding:"16px",color:"var(--text-muted)",fontSize:"0.8rem"},children:"No active jobs"})]})]})]})}function ul({job:e,status:t,onCancel:n}){const s={running:"#22c55e",pending:"#fbbf24",completed:"#3b82f6"},a={running:Oe,pending:Ra,completed:Xm}[t];return r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px",padding:"6px 8px",backgroundColor:"var(--bg-input)",borderRadius:"4px",marginBottom:"4px",fontSize:"0.8rem"},children:[r.jsx(a,{size:12,color:s[t],className:t==="running"?"spin":""}),r.jsxs("div",{style:{flex:1,minWidth:0},children:[r.jsx("div",{style:{whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis",fontWeight:500},children:e.prompt||e.prompt_id.slice(0,8)}),r.jsxs("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)"},children:[e.resolution," ",e.aspect_ratio," ",e.num_frames&&`• ${e.num_frames}f`]})]}),t!=="completed"&&n&&r.jsx("button",{onClick:()=>n(e.prompt_id),style:{background:"transparent",border:"none",cursor:"pointer",padding:"2px"},children:r.jsx(It,{size:12,color:"var(--text-muted)"})}),t==="completed"&&e.output_video&&r.jsx("a",{href:`${ee}${e.output_video}`,target:"_blank",rel:"noopener noreferrer",style:{color:"#3b82f6",fontSize:"0.7rem"},children:"View"})]})}async function Nt(e,t,n={}){const s=await fetch(e,{method:"POST",body:t,headers:n,credentials:"same-origin"}),a=await s.text();try{const l=a?JSON.parse(a):null;return{ok:s.ok,status:s.status,data:l}}catch{return{ok:s.ok,status:s.status,data:a}}}async function Zu(e){const t=await fetch(e,{method:"GET",credentials:"same-origin"}),n=await t.text();try{const s=n?JSON.parse(n):null;return{ok:t.ok,status:t.status,data:s}}catch{return{ok:t.ok,status:t.status,data:n}}}async function xx(e,t={}){const n=await fetch(e,{method:"POST",body:JSON.stringify(t),headers:{"Content-Type":"application/json"},credentials:"same-origin"}),s=await n.text();try{const a=s?JSON.parse(s):null;return{ok:n.ok,status:n.status,data:a}}catch{return{ok:n.ok,status:n.status,data:s}}}const gx=[{value:"480p",label:"480p",desc:"Fast"},{value:"720p",label:"720p",desc:"Balanced"}],vx=[8,12,16,24],yx=["16:9","9:16","1:1","4:3","3:4"];function jx({onOutput:e,onRefreshHistory:t,onJobSubmitted:n}){const[s,a]=c.useState(()=>localStorage.getItem("t2v_prompt")||""),[l,o]=c.useState("blurry, low quality, distorted, ugly"),[i,d]=c.useState(41),[u,y]=c.useState("1:1"),[g,x]=c.useState("480p"),[k,S]=c.useState(16),[z,R]=c.useState(!1),[f,p]=c.useState(6),[m,h]=c.useState(1),[j,_]=c.useState(-1),[P,I]=c.useState(20),[G,H]=c.useState(6),[N,C]=c.useState(!1),[L,X]=c.useState(""),[A,$]=c.useState(""),[O,M]=c.useState(0);c.useRef(null);const B=T=>{a(T),localStorage.setItem("t2v_prompt",T)},Q=c.useMemo(()=>s.trim().length>0&&!N,[s,N]),te=async(T,v=180)=>{for(let K=0;K<v;K++){await new Promise(b=>setTimeout(b,1e3));try{const b=await fetch(`${ee}/comfyui/job/${T}`);if(!b.ok)continue;const D=await b.json();if(D.status==="pending")$("Queued..."),M(Math.min(10,K));else if(D.status==="running")$("Generating..."),M(Math.min(90,10+K));else{if(D.status==="completed")return M(100),$("Done!"),D;if(D.status==="failed")throw new Error(D.error||"Generation failed")}}catch(b){if(b.message.includes("failed"))throw b}}throw new Error("Generation timed out")},oe=async()=>{var v,K,b;if(!s.trim()){X("Prompt is required");return}C(!0),X(""),$("Submitting..."),M(0);const T=new FormData;T.append("prompt",s),T.append("num_frames",String(i)),T.append("aspect_ratio",u),T.append("resolution",g),T.append("fps",String(k));try{const D=await Nt(`${ee}/generate-text`,T);if(!D.ok)throw new Error(((v=D.data)==null?void 0:v.detail)||`Generation failed (status ${D.status})`);const V=(K=D.data)==null?void 0:K.prompt_id;if(!V)throw new Error("No prompt_id returned");$("Queued..."),n&&n();const Y=await te(V);if(Y.output_video||Y.url){const F=Y.output_video||Y.url,le=F.startsWith("http")?F:`${ee}${F}`;e({kind:"video",url:le,backendUrl:le,filename:F.split("/").pop(),meta:{...(b=D.data)==null?void 0:b.meta,prompt_id:V}}),t&&t()}}catch(D){const V=(D==null?void 0:D.message)||"Failed to generate video";X(V),await La({level:"error",message:"Text-to-video failed",timestamp:new Date().toISOString(),meta:{message:V}})}finally{C(!1),$(""),M(0)}};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(os,{size:18}),"Video Prompt"]}),r.jsx("textarea",{className:"prompt-textarea",value:s,onChange:T=>B(T.target.value),rows:4,placeholder:"Describe the video you want to generate... (e.g., 'a cat walking through a field of flowers, cinematic')"}),r.jsxs("div",{className:"char-count",children:[s.length," characters"]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Settings"}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Resolution"}),r.jsx("div",{className:"button-group",children:gx.map(T=>r.jsx("button",{className:`btn-option ${g===T.value?"active":""}`,onClick:()=>x(T.value),type:"button",children:T.label},T.value))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Aspect Ratio"}),r.jsx("div",{className:"button-group",children:yx.map(T=>r.jsx("button",{className:`btn-option ${u===T?"active":""}`,onClick:()=>y(T),type:"button",children:T},T))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Frame Rate"}),r.jsx("div",{className:"button-group",children:vx.map(T=>r.jsxs("button",{className:`btn-option ${k===T?"active":""}`,onClick:()=>S(T),type:"button",children:[T," fps"]},T))})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("label",{children:["Duration",r.jsxs("span",{className:"label-value",children:[(i/k).toFixed(1),"s (",i," frames)"]})]}),r.jsx("input",{type:"range",min:"17",max:"81",step:"4",value:i,onChange:T=>d(parseInt(T.target.value,10))}),r.jsxs("div",{className:"range-labels",children:[r.jsxs("span",{children:[(17/k).toFixed(1),"s"]}),r.jsxs("span",{children:[(81/k).toFixed(1),"s"]})]})]})]}),r.jsxs("div",{className:"tool-section collapsible",children:[r.jsxs("button",{className:"section-toggle",onClick:()=>R(!z),children:[r.jsx(hr,{size:16}),"Advanced Settings",r.jsx(Mt,{size:16,className:z?"rotated":""})]}),z&&r.jsxs("div",{className:"advanced-content",children:[r.jsxs("div",{className:"form-row",children:[r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"Video Steps"}),r.jsx("input",{type:"number",value:f,onChange:T=>p(parseInt(T.target.value)||6),min:"1",max:"30"})]}),r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"Video CFG"}),r.jsx("input",{type:"number",value:m,onChange:T=>h(parseFloat(T.target.value)||1),min:"0.1",max:"10",step:"0.1"})]})]}),r.jsxs("div",{className:"form-row",children:[r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"T2I Steps"}),r.jsx("input",{type:"number",value:P,onChange:T=>I(parseInt(T.target.value)||20),min:"1",max:"50"})]}),r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"T2I CFG"}),r.jsx("input",{type:"number",value:G,onChange:T=>H(parseFloat(T.target.value)||6),min:"1",max:"20",step:"0.5"})]})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Seed (-1 = random)"}),r.jsx("input",{type:"number",value:j,onChange:T=>_(parseInt(T.target.value)||-1)})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Negative Prompt"}),r.jsx("textarea",{value:l,onChange:T=>o(T.target.value),rows:2,placeholder:"Things to avoid..."})]})]})]}),N&&r.jsxs("div",{className:"progress-section",children:[r.jsx("div",{className:"progress-bar",children:r.jsx("div",{className:"progress-fill",style:{width:`${O}%`}})}),r.jsxs("div",{className:"progress-status",children:[r.jsx(Oe,{size:16,className:"spin"}),A]})]}),L&&r.jsxs("div",{className:"error-message",children:["⚠️ ",L]}),r.jsx("button",{className:"btn-primary btn-large",type:"button",disabled:!Q,onClick:oe,children:N?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Generating..."]}):r.jsxs(r.Fragment,{children:[r.jsx(os,{size:18}),"Generate Video"]})}),r.jsx("div",{className:"tool-info",children:"💡 Text-to-Video first generates an image from your prompt, then animates it using Wan2.2"}),r.jsx("style",{children:`
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
      `})]})}function bx({onPresetChange:e,onParametersChange:t,currentParameters:n}){var P,I,G,H;const[s,a]=c.useState([]),[l,o]=c.useState(null),[i,d]=c.useState({}),[u,y]=c.useState(!0),[g,x]=c.useState(!0),[k,S]=c.useState(null);c.useEffect(()=>{z()},[]);const z=async()=>{var N;try{x(!0);const C=await fetch(`${ee}/api/presets`);if(!C.ok)throw new Error("Failed to fetch presets");const L=await C.json();if(a(L.presets||[]),((N=L.presets)==null?void 0:N.length)>0){const X=L.presets[0];o(X),R(X)}}catch(C){console.error("Failed to load presets:",C),S(C.message),a(wx())}finally{x(!1)}},R=N=>{if(!(N!=null&&N.parameters))return;const C={};Object.entries(N.parameters).forEach(([L,X])=>{X.type!=="image"&&(C[L]=X.default??X.value??"")}),d(C),t==null||t(C)},f=N=>{o(N),R(N),e==null||e(N)},p=(N,C,L)=>{let X=C;L.type==="integer"?X=parseInt(C,10):L.type==="float"&&(X=parseFloat(C));const A={...i,[N]:X};d(A),t==null||t(A)},m=N=>{switch(N){case"ImageToVideo":return r.jsx(ls,{size:16});case"TextToVideo":return r.jsx(Ut,{size:16});case"TextToImage":return r.jsx(Ju,{size:16});default:return r.jsx(hr,{size:16})}},h=N=>{var C,L,X,A,$,O;return(C=N.name)!=null&&C.toLowerCase().includes("lightning")||(L=N.name)!=null&&L.toLowerCase().includes("fast")?r.jsx("span",{className:"preset-badge fast",children:"⚡ Fast"}):(X=N.name)!=null&&X.toLowerCase().includes("quality")||(A=N.name)!=null&&A.toLowerCase().includes("q6")?r.jsx("span",{className:"preset-badge quality",children:"💎 Quality"}):($=N.name)!=null&&$.toLowerCase().includes("nsfw")||(O=N.name)!=null&&O.toLowerCase().includes("enhanced")?r.jsx("span",{className:"preset-badge nsfw",children:"🔥 Enhanced"}):null},j=(N,C)=>{var X;const L=i[N]??C.default??"";return C.type==="image"?null:C.type==="string"?r.jsxs("div",{className:"param-group",children:[r.jsxs("label",{htmlFor:`param-${N}`,children:[C.label||N,C.description&&r.jsx("span",{className:"param-hint",title:C.description,children:"ℹ️"})]}),r.jsx("textarea",{id:`param-${N}`,value:L,onChange:A=>p(N,A.target.value,C),placeholder:C.description,rows:N.includes("prompt")?3:1})]},N):C.type==="integer"&&C.min!==void 0&&C.max!==void 0?r.jsxs("div",{className:"param-group",children:[r.jsxs("label",{htmlFor:`param-${N}`,children:[C.label||N,": ",r.jsx("span",{className:"param-value",children:L}),C.description&&r.jsx("span",{className:"param-hint",title:C.description,children:"ℹ️"})]}),r.jsx("input",{id:`param-${N}`,type:"range",min:C.min,max:C.max,step:C.step||1,value:L,onChange:A=>p(N,A.target.value,C)}),r.jsxs("div",{className:"range-labels",children:[r.jsx("span",{children:C.min}),r.jsx("span",{children:C.max})]})]},N):C.type==="float"&&C.min!==void 0&&C.max!==void 0?r.jsxs("div",{className:"param-group",children:[r.jsxs("label",{htmlFor:`param-${N}`,children:[C.label||N,": ",r.jsx("span",{className:"param-value",children:((X=L.toFixed)==null?void 0:X.call(L,2))||L}),C.description&&r.jsx("span",{className:"param-hint",title:C.description,children:"ℹ️"})]}),r.jsx("input",{id:`param-${N}`,type:"range",min:C.min,max:C.max,step:C.step||.1,value:L,onChange:A=>p(N,A.target.value,C)}),r.jsxs("div",{className:"range-labels",children:[r.jsx("span",{children:C.min}),r.jsx("span",{children:C.max})]})]},N):C.type==="integer"||C.type==="float"?r.jsxs("div",{className:"param-group",children:[r.jsxs("label",{htmlFor:`param-${N}`,children:[C.label||N,C.description&&r.jsx("span",{className:"param-hint",title:C.description,children:"ℹ️"})]}),r.jsx("input",{id:`param-${N}`,type:"number",value:L,onChange:A=>p(N,A.target.value,C),step:C.step||(C.type==="float"?.1:1)})]},N):C.type==="boolean"?r.jsx("div",{className:"param-group checkbox",children:r.jsxs("label",{htmlFor:`param-${N}`,children:[r.jsx("input",{id:`param-${N}`,type:"checkbox",checked:!!L,onChange:A=>p(N,A.target.checked,C)}),C.label||N,C.description&&r.jsx("span",{className:"param-hint",title:C.description,children:"ℹ️"})]})},N):null},_=()=>{if(!(l!=null&&l.parameters))return{};const N={prompt:[],generation:[],dimensions:[],other:[]};return Object.entries(l.parameters).forEach(([C,L])=>{L.type!=="image"&&(C.includes("prompt")?N.prompt.push([C,L]):["steps","cfg","seed","frame_rate"].includes(C)?N.generation.push([C,L]):["width","height","num_frames"].includes(C)?N.dimensions.push([C,L]):N.other.push([C,L]))}),N};return g?r.jsxs("div",{className:"preset-selector loading",children:[r.jsx(ga,{className:"spinning",size:24}),r.jsx("span",{children:"Loading presets..."})]}):r.jsxs("div",{className:"preset-selector",children:[r.jsxs("div",{className:"preset-header",onClick:()=>y(!u),children:[r.jsxs("div",{className:"preset-title",children:[r.jsx(ga,{size:20}),r.jsx("span",{children:"Workflow Preset"}),l&&r.jsx("span",{className:"selected-preset-name",children:l.name})]}),u?r.jsx(Wm,{size:20}):r.jsx(Mt,{size:20})]}),u&&r.jsxs("div",{className:"preset-content",children:[r.jsx("div",{className:"preset-list",children:s.map(N=>r.jsxs("div",{className:`preset-card ${(l==null?void 0:l.id)===N.id?"selected":""}`,onClick:()=>f(N),children:[r.jsxs("div",{className:"preset-card-header",children:[m(N.category),r.jsx("span",{className:"preset-name",children:N.name}),h(N)]}),r.jsx("p",{className:"preset-description",children:N.description})]},N.id))}),l&&r.jsxs("div",{className:"preset-parameters",children:[r.jsxs("h4",{children:[r.jsx(hr,{size:16})," Parameters"]}),((P=_().prompt)==null?void 0:P.length)>0&&r.jsxs("div",{className:"param-section",children:[r.jsx("h5",{children:"📝 Prompts"}),_().prompt.map(([N,C])=>j(N,C))]}),((I=_().generation)==null?void 0:I.length)>0&&r.jsxs("div",{className:"param-section",children:[r.jsx("h5",{children:"⚙️ Generation"}),r.jsx("div",{className:"param-grid",children:_().generation.map(([N,C])=>j(N,C))})]}),((G=_().dimensions)==null?void 0:G.length)>0&&r.jsxs("div",{className:"param-section",children:[r.jsx("h5",{children:"📐 Dimensions"}),r.jsx("div",{className:"param-grid",children:_().dimensions.map(([N,C])=>j(N,C))})]}),((H=_().other)==null?void 0:H.length)>0&&r.jsxs("div",{className:"param-section",children:[r.jsx("h5",{children:"🔧 Other"}),_().other.map(([N,C])=>j(N,C))]})]})]}),k&&r.jsxs("div",{className:"preset-error",children:["⚠️ ",k," - Using default presets"]})]})}function wx(){return[{id:"wan22_enhanced_q4km",name:"WAN 2.2 Enhanced NSFW FastMove",category:"ImageToVideo",description:"Lightning-fast I2V with NSFW FastMove LoRAs. 4 steps, cfg=1.",parameters:{prompt:{type:"string",default:"motion, smooth camera movement",label:"Prompt"},steps:{type:"integer",default:4,min:2,max:12,label:"Steps"},cfg:{type:"float",default:1,min:1,max:3,step:.1,label:"CFG Scale"},seed:{type:"integer",default:-1,label:"Seed",description:"-1 for random"},width:{type:"integer",default:480,min:256,max:1280,step:16,label:"Width"},height:{type:"integer",default:480,min:256,max:1280,step:16,label:"Height"},num_frames:{type:"integer",default:41,min:17,max:81,step:8,label:"Frames"}}},{id:"wan22_q6_quality",name:"WAN 2.2 Q6 Quality",category:"ImageToVideo",description:"Higher quality 6-bit model with DPM++ scheduler. Best visual quality.",parameters:{prompt:{type:"string",default:"cinematic motion",label:"Prompt"},steps:{type:"integer",default:8,min:4,max:20,label:"Steps"},cfg:{type:"float",default:2.5,min:1,max:5,step:.1,label:"CFG Scale"},seed:{type:"integer",default:-1,label:"Seed"},width:{type:"integer",default:512,min:256,max:1280,step:16,label:"Width"},height:{type:"integer",default:512,min:256,max:1280,step:16,label:"Height"},num_frames:{type:"integer",default:49,min:17,max:97,step:8,label:"Frames"}}}]}const kx=[8,12,16,24],Sx=[{value:"wan2.2",label:"🎬 Wan2.2 14B Q6 DisTorch2",desc:"High quality via ComfyUI"}],Nx={"480p":{label:"480p",dimensions:{"16:9":"848×480","9:16":"480×848","1:1":"480×480","4:3":"640×480","3:4":"480×640"}},"576p":{label:"576p",dimensions:{"16:9":"1024×576","9:16":"576×1024","1:1":"576×576","4:3":"768×576","3:4":"576×768"}},"720p":{label:"720p",dimensions:{"16:9":"1280×720","9:16":"720×1280","1:1":"720×720","4:3":"960×720","3:4":"720×960"}}},Cx=["16:9","9:16","1:1","4:3","3:4"];function _x({onOutput:e,onRefreshHistory:t,onCreationsModeChange:n,onParamsChange:s,onJobSubmitted:a}){var ae,ve,ce,he,Yt;const l=c.useRef(null),[o,i]=c.useState(null),[d,u]=c.useState(""),[y,g]=c.useState("file"),[x,k]=c.useState(()=>{try{return localStorage.getItem("oelala_last_prompt")||""}catch{return""}}),[S,z]=c.useState("low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, unnatural sand behavior, visual glitches"),[R,f]=c.useState(!1),[p,m]=c.useState(!1),[h,j]=c.useState(6),[_,P]=c.useState("480p"),[I,G]=c.useState("wan2.2"),[H,N]=c.useState("v2"),[C,L]=c.useState(!1),[X,A]=c.useState("9:16"),[$,O]=c.useState(16),[M,B]=c.useState(6),[Q,te]=c.useState(1),[oe,T]=c.useState(-1),[v,K]=c.useState(!1),[b,D]=c.useState({high_noise:[],low_noise:[],general:[]}),[V,Y]=c.useState([]),[F,le]=c.useState(!1),[re,ie]=c.useState({high_noise:[],low_noise:[],pairs:[]}),[me,Ze]=c.useState("wan2.2_i2v_high_noise_14B_Q6_K.gguf"),[Re,ht]=c.useState("wan2.2_i2v_low_noise_14B_Q6_K.gguf"),[Xt,Lr]=c.useState(!1),[et,yr]=c.useState(!1),[Ct,xe]=c.useState(1),[mn,fs]=c.useState(!1),[hn,ms]=c.useState(null),[Fa,hs]=c.useState({}),[jr,xn]=c.useState(!1),[gn,xt]=c.useState(""),[Da,Fr]=c.useState(null),xs=c.useMemo(()=>!!o&&!jr,[o,jr]);c.useEffect(()=>{(async()=>{try{const Z=await fetch(`${ee}/loras`);if(Z.ok){const de=await Z.json();D(de)}}catch(Z){console.error("Failed to fetch LoRAs:",Z)}})()},[]),c.useEffect(()=>{(async()=>{try{const Z=await fetch(`${ee}/unet-models`);if(Z.ok){const de=await Z.json();ie(de)}}catch(Z){console.error("Failed to fetch Unet models:",Z)}})()},[]),c.useEffect(()=>{if(x)try{localStorage.setItem("oelala_last_prompt",x)}catch{}},[x]),c.useEffect(()=>{s&&s({tool:"ImageToVideo",prompt:x,duration:h,resolution:_,modelMode:I,modelVersion:H,aspectRatio:X,fps:$,steps:M,cfg:Q,seed:oe,usePose:C,loraConfigs:V,filename:(o==null?void 0:o.name)||null})},[x,h,_,I,H,X,$,M,Q,oe,C,V,o,s]);const vn=c.useCallback(async E=>{Fr(E),xt("");try{const Z=`${ee}${E.url}`,pe=await(await fetch(Z)).blob(),Be=E.filename||E.url.split("/").pop(),tt=new File([pe],Be,{type:pe.type||"image/png"});i(tt),u(Z),g("file"),e({kind:"image",url:Z,backendUrl:Z,filename:Be,meta:{source:"my-creations",originalItem:E}})}catch(Z){xt("Failed to load selected image"),console.error("Error selecting creation:",Z)}},[e]);c.useEffect(()=>(n&&n(y==="creations"&&!o,vn),()=>{n&&n(!1,null)}),[y,o,n,vn]);const w=async E=>{if(!E)return;i(E),xt(""),Fr(null);const Z=URL.createObjectURL(E);u(Z);try{const de=new FormData;de.append("file",E);const pe=await fetch(`${ee}/extract-metadata`,{method:"POST",body:de});if(pe.ok){const Be=await pe.json();Be.prompt&&!x&&k(Be.prompt),Be.negative_prompt&&S==="low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, unnatural sand behavior, visual glitches"&&z(Be.negative_prompt)}}catch{}},W=()=>{i(null),u(""),Fr(null),l.current&&(l.current.value="")},J=async()=>{var Be,tt,ai,li;if(!o){xt("Image is required");return}xn(!0),xt("");const E=h*$,Z=new FormData;Z.append("file",o),Z.append("num_frames",String(E)),Z.append("resolution",_),Z.append("fps",String($)),Z.append("aspect_ratio",X),C||Z.append("prompt",x||"Motion, subject moving naturally");let de,pe=!0;C?(de=`${ee}/generate-pose`,pe=!1):(de=`${ee}/generate-wan22-async`,Z.append("steps",String(M)),Z.append("cfg",String(Q)),Z.append("seed",String(oe)),et&&Ct>1&&(Z.append("extend_mode","true"),Z.append("clip_count",String(Ct))),me&&Z.append("unet_high_noise",me),Re&&Z.append("unet_low_noise",Re),V.length>0&&Z.append("lora_configs",JSON.stringify(V)));try{const it=await Nt(de,Z);if(!it.ok){xt(((Be=it.data)==null?void 0:Be.detail)||`Generation failed (status ${it.status})`);return}if(pe)a&&a(it.data);else{const yn=((tt=it.data)==null?void 0:tt.video_url)||((ai=it.data)==null?void 0:ai.url),rp=(li=it.data)==null?void 0:li.output_video,oi=yn?`${ee}${yn}`:"";e({kind:"video",url:oi,backendUrl:oi,filename:rp,meta:it.data}),t()}}catch(it){const yn=(it==null?void 0:it.message)||"Failed to generate video";xt(yn),await La({level:"error",message:"Image-to-video failed",timestamp:new Date().toISOString(),meta:{message:yn,modelMode:I}})}finally{xn(!1)}};return r.jsxs("div",{className:"tool-container",children:[r.jsx("style",{children:`
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}),r.jsxs("div",{className:"grok-card",children:[r.jsx("div",{className:"grok-card-header",children:r.jsx("div",{className:"grok-card-title",children:"Model Selection"})}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Generation Mode"}),r.jsxs("div",{style:{position:"relative"},children:[r.jsx("select",{value:I,onChange:E=>{G(E.target.value),E.target.value==="wan2.2"&&(P("576p"),A("9:16"),j(6))},style:{width:"100%",padding:"12px 40px 12px 16px",backgroundColor:"var(--bg-secondary, #1a1a1a)",border:"1px solid var(--border-color)",borderRadius:"8px",color:"var(--text-primary, #fff)",fontSize:"1rem",appearance:"none",cursor:"pointer"},children:Sx.map(E=>r.jsx("option",{value:E.value,style:{backgroundColor:"#1a1a1a",color:"#fff"},children:E.label},E.value))}),r.jsx(Mt,{size:20,style:{position:"absolute",right:"12px",top:"50%",transform:"translateY(-50%)",pointerEvents:"none",color:"var(--text-muted)"}})]}),r.jsxs("div",{className:"info-badge",style:{marginTop:"8px"},children:[r.jsx("span",{style:{fontWeight:600},children:"🎬 Wan2.2 14B Q6"})," • ",r.jsx("span",{style:{color:"#93c5fd"},children:"ComfyUI Backend"}),r.jsx("div",{style:{marginTop:"4px",opacity:.8},children:"High-quality I2V with DisTorch2 + SageAttention (10GB VRAM)"})]})]}),r.jsxs("div",{style:{marginTop:"12px",paddingTop:"12px",borderTop:"1px solid var(--border-color)"},children:[r.jsxs("div",{onClick:()=>Lr(!Xt),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer",padding:"4px 0"},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[r.jsx(Uh,{size:16}),r.jsx("span",{style:{fontSize:"0.9rem",fontWeight:500},children:"Unet Model"}),r.jsxs("span",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:["(",me.replace(".gguf","").replace("wan2.2_i2v_",""),")"]})]}),r.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:Xt?"▼":"▶"})]}),Xt&&r.jsxs("div",{style:{marginTop:"12px",display:"flex",flexDirection:"column",gap:"12px"},children:[r.jsxs("div",{children:[r.jsx("label",{style:{display:"block",fontSize:"0.8rem",color:"var(--text-muted)",marginBottom:"4px"},children:"Model Pair (recommended)"}),r.jsx("select",{onChange:E=>{var de;const Z=(de=re.pairs)==null?void 0:de.find(pe=>pe.name===E.target.value);Z&&(Ze(Z.high.path),ht(Z.low.path))},style:{width:"100%",padding:"8px 12px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"6px",color:"var(--text-primary)",fontSize:"0.85rem"},value:((ve=(ae=re.pairs)==null?void 0:ae.find(E=>E.high.path===me&&E.low.path===Re))==null?void 0:ve.name)||"",children:(ce=re.pairs)==null?void 0:ce.map(E=>r.jsxs("option",{value:E.name,children:[E.name," (",E.high.size_gb,"GB)"]},E.name))})]}),r.jsxs("details",{style:{fontSize:"0.8rem"},children:[r.jsx("summary",{style:{cursor:"pointer",color:"var(--text-muted)",marginBottom:"8px"},children:"⚙️ Advanced: Select models separately"}),r.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"12px",paddingTop:"8px"},children:[r.jsxs("div",{children:[r.jsx("label",{style:{display:"block",fontSize:"0.8rem",color:"var(--text-muted)",marginBottom:"4px"},children:"High Noise Model (steps 0-3)"}),r.jsx("select",{value:me,onChange:E=>Ze(E.target.value),style:{width:"100%",padding:"8px 12px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"6px",color:"var(--text-primary)",fontSize:"0.85rem"},children:(he=re.high_noise)==null?void 0:he.map(E=>r.jsxs("option",{value:E.path,children:[E.name," (",E.size_gb,"GB)"]},E.path))})]}),r.jsxs("div",{children:[r.jsx("label",{style:{display:"block",fontSize:"0.8rem",color:"var(--text-muted)",marginBottom:"4px"},children:"Low Noise Model (steps 3+)"}),r.jsx("select",{value:Re,onChange:E=>ht(E.target.value),style:{width:"100%",padding:"8px 12px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"6px",color:"var(--text-primary)",fontSize:"0.85rem"},children:(Yt=re.low_noise)==null?void 0:Yt.map(E=>r.jsxs("option",{value:E.path,children:[E.name," (",E.size_gb,"GB)"]},E.path))})]})]})]})]})]})]}),r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsxs("div",{className:"grok-card-title",style:{display:"flex",alignItems:"center",gap:"6px"},children:["Positive Prompt ",r.jsx("span",{style:{fontWeight:400,color:"var(--text-muted)",fontSize:"0.85rem"},children:"(Describe the motion)"}),r.jsxs("div",{style:{position:"relative",display:"inline-block"},children:[r.jsx("button",{className:"icon-btn",style:{width:"20px",height:"20px",border:"none",background:"transparent",padding:0},onClick:()=>m(!p),title:"Prompt tips",children:r.jsx(Hu,{size:14,color:p?"#fbbf24":"#666666"})}),p&&r.jsxs("div",{style:{position:"absolute",top:"100%",left:"50%",transform:"translateX(-50%)",marginTop:"8px",backgroundColor:"#1a1a1a",border:"1px solid #fbbf24",borderRadius:"8px",padding:"12px",width:"280px",zIndex:100,fontSize:"0.8rem",color:"#fbbf24",boxShadow:"0 4px 12px rgba(0,0,0,0.5)"},children:[r.jsx("div",{style:{fontWeight:600,marginBottom:"8px"},children:"💡 Prompt Tips"}),r.jsxs("ul",{style:{margin:0,paddingLeft:"16px",lineHeight:1.6},children:[r.jsx("li",{children:"Structure: [subject + motion] + [scene] + [camera]"}),r.jsx("li",{children:'Focus on motion - "walking slowly", "hair blowing"'}),r.jsx("li",{children:'Add intensity - "quickly", "gently", "dramatically"'}),r.jsx("li",{children:'Camera moves - "slow zoom in", "pan left"'}),r.jsx("li",{children:"Describe what you want, not what to avoid"})]})]})]})]}),r.jsxs("div",{style:{display:"flex",gap:"4px"},children:[r.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px"},onClick:async()=>{if(d)try{const Z=await(await fetch(`${ee}/extract-metadata-url`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image_url:d})})).json();Z.positive_prompt&&k(Z.positive_prompt),Z.negative_prompt&&setNegPrompt(Z.negative_prompt)}catch(E){console.error("Extract metadata failed:",E)}},title:"Extract prompt from selected image",disabled:!d,children:r.jsx(lh,{size:14,color:d?"#fbbf24":"#666666"})}),r.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px"},children:r.jsx(Ku,{size:14,color:"#fbbf24"})}),r.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px"},children:r.jsx(Ut,{size:14,color:"#fbbf24"})})]})]}),r.jsxs("div",{style:{position:"relative"},children:[r.jsx("textarea",{className:"form-textarea",value:x,onChange:E=>k(E.target.value),rows:4,placeholder:"Describe how you want the image to move or animate... (Optional for image-to-video)",style:{backgroundColor:"#0f0f0f",border:"1px solid var(--border-color)",borderRadius:"8px",resize:"vertical",minHeight:"80px",padding:"12px",paddingBottom:"28px",width:"100%",boxSizing:"border-box"}}),r.jsxs("div",{style:{position:"absolute",bottom:"8px",right:"8px",fontSize:"0.7rem",color:"var(--text-muted)"},children:[x.length,"/2048"]})]}),r.jsxs("div",{style:{marginTop:"12px"},children:[r.jsxs("div",{onClick:()=>f(!R),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer",padding:"8px 0"},children:[r.jsx("span",{style:{fontSize:"0.85rem",color:"var(--text-secondary)"},children:"Negative Prompt"}),r.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:R?"▼":"▶"})]}),R&&r.jsxs("div",{style:{position:"relative"},children:[r.jsx("textarea",{className:"form-textarea",value:S,onChange:E=>z(E.target.value),rows:3,placeholder:"Things to avoid in the generation...",style:{backgroundColor:"#0f0f0f",border:"1px solid var(--border-color)",borderRadius:"8px",resize:"vertical",minHeight:"60px",padding:"12px",paddingBottom:"28px",width:"100%",boxSizing:"border-box",fontSize:"0.85rem"}}),r.jsxs("div",{style:{position:"absolute",bottom:"8px",right:"8px",fontSize:"0.7rem",color:"var(--text-muted)"},children:[S.length,"/2048"]})]})]})]}),r.jsxs("div",{className:"grok-card",children:[r.jsx("div",{className:"grok-card-header",children:r.jsx("div",{className:"grok-card-title",children:"Upload Photo"})}),r.jsxs("div",{className:"grok-tabs",children:[r.jsxs("button",{className:`grok-tab ${y==="file"?"active":""}`,onClick:()=>g("file"),children:[r.jsx(St,{size:14})," Upload File"]}),r.jsxs("button",{className:`grok-tab ${y==="url"?"active":""}`,onClick:()=>g("url"),children:[r.jsx(wh,{size:14})," From URL"]}),r.jsxs("button",{className:`grok-tab ${y==="creations"?"active":""}`,onClick:()=>g("creations"),children:[r.jsx(Qu,{size:14})," From My Creations"]})]}),r.jsx("input",{ref:l,type:"file",accept:"image/*",onChange:E=>{var Z;return w((Z=E.target.files)==null?void 0:Z[0])},style:{display:"none"}}),y==="file"&&!o&&r.jsxs("div",{className:"upload-box",onClick:()=>{var E;return(E=l.current)==null?void 0:E.click()},style:{cursor:"pointer",borderStyle:"dashed",minHeight:"200px",justifyContent:"center"},children:[r.jsx(St,{size:48,className:"text-muted",style:{opacity:.2}}),r.jsx("div",{style:{fontSize:"1rem",fontWeight:500,color:"var(--text-secondary)"},children:"Drag & drop an image here, or click to browse"}),r.jsx("div",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:"JPEG, PNG, WebP, Max 20MB"}),r.jsx("div",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:"Minimum size: 300x300px"})]}),y==="url"&&!o&&r.jsxs("div",{style:{padding:"16px 0"},children:[r.jsx("div",{style:{fontSize:"0.85rem",color:"var(--text-muted)",marginBottom:"8px"},children:"Enter image URL:"}),r.jsx("input",{type:"url",placeholder:"https://example.com/image.jpg",style:{width:"100%",padding:"12px",background:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"8px",color:"var(--text-primary)",fontSize:"0.9rem"},onKeyDown:async E=>{if(E.key==="Enter"&&E.target.value)try{const de=await(await fetch(E.target.value)).blob(),pe=E.target.value.split("/").pop()||"image.jpg",Be=new File([de],pe,{type:de.type});w(Be)}catch{xt("Failed to load image from URL")}}}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"8px"},children:"Press Enter to load"})]}),y==="creations"&&!o&&r.jsxs("div",{style:{padding:"24px 16px",textAlign:"center",color:"var(--text-muted)",backgroundColor:"var(--bg-secondary)",borderRadius:"8px",border:"1px dashed var(--border-color)"},children:[r.jsx(mr,{size:32,style:{opacity:.5,marginBottom:"12px"}}),r.jsx("div",{style:{fontSize:"0.9rem",marginBottom:"8px"},children:"Select an image from the panel on the right →"}),r.jsx("div",{style:{fontSize:"0.8rem",opacity:.7},children:"Browse your generated images"})]}),o&&r.jsxs("div",{className:"relative",style:{position:"relative"},children:[r.jsx("img",{src:d,alt:"Preview",style:{width:"100%",maxHeight:"400px",objectFit:"contain",borderRadius:"8px",border:"1px solid var(--border-color)"}}),r.jsx("button",{onClick:E=>{E.stopPropagation(),W()},style:{position:"absolute",top:"12px",right:"12px",background:"rgba(0,0,0,0.7)",border:"none",color:"white",borderRadius:"50%",width:"32px",height:"32px",display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",backdropFilter:"blur(4px)"},children:r.jsx(It,{size:18})})]})]}),r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"form-group",children:[r.jsxs("label",{className:"grok-section-label",children:["Resolution",r.jsx("span",{className:"text-muted",style:{fontWeight:400},children:" (Higher = Better Quality, more VRAM)"})]}),r.jsx("div",{className:"grok-toggle-group",children:Object.entries(Nx).map(([E,Z])=>r.jsxs("button",{className:`grok-toggle-btn ${_===E?"active":""}`,onClick:()=>P(E),children:[Z.label,r.jsx("span",{style:{fontSize:"0.7rem",opacity:.7,display:"block"},children:Z.dimensions[X]||Z.dimensions["1:1"]})]},E))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Aspect Ratio"}),r.jsx("div",{className:"grok-toggle-group",children:Cx.map(E=>r.jsx("button",{className:`grok-toggle-btn ${X===E?"active":""}`,onClick:()=>A(E),children:E},E))})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"8px"},children:[r.jsx("label",{className:"grok-section-label",children:"Duration"}),r.jsxs("span",{className:"nav-badge",style:{fontSize:"0.8rem"},children:[h,"s (",h*$,"f)"]})]}),r.jsxs("div",{style:{position:"relative",height:"24px",marginBottom:"8px"},children:[r.jsx("input",{type:"range",min:"3",max:"15",step:"1",value:h,onChange:E=>j(parseInt(E.target.value,10)),style:{width:"100%",opacity:0,position:"absolute",zIndex:2,cursor:"pointer"}}),r.jsx("div",{style:{position:"absolute",top:"10px",left:0,right:0,height:"4px",backgroundColor:"#333",borderRadius:"2px"},children:r.jsx("div",{style:{width:`${(h-3)/12*100}%`,height:"100%",backgroundColor:"var(--accent-color, #a855f7)",borderRadius:"2px"}})}),r.jsx("div",{style:{position:"absolute",top:"2px",left:`calc(${(h-3)/12*100}% - 10px)`,width:"20px",height:"20px",backgroundColor:"white",borderRadius:"50%",boxShadow:"0 2px 4px rgba(0,0,0,0.3)"}})]}),r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.75rem",color:"var(--text-muted)"},children:[r.jsx("span",{children:"3s"}),r.jsx("span",{children:"6s (rec)"}),r.jsx("span",{children:"15s"})]})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"8px"},children:[r.jsx("label",{className:"grok-section-label",children:"Frame Rate (FPS)"}),r.jsxs("span",{className:"nav-badge",style:{fontSize:"0.8rem"},children:[$," fps"]})]}),r.jsx("div",{className:"grok-toggle-group",children:kx.map(E=>r.jsx("button",{className:`grok-toggle-btn ${$===E?"active":""}`,onClick:()=>O(E),type:"button",children:E},E))}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"8px"},children:"Higher FPS = smoother motion, more VRAM required"})]}),I!=="wan2.2"&&r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Model Version"}),r.jsxs("div",{className:"grok-toggle-group",children:[r.jsx("button",{className:`grok-toggle-btn ${H==="v1"?"active":""}`,onClick:()=>N("v1"),children:"V1"}),r.jsx("button",{className:`grok-toggle-btn ${H==="v2"?"active":""}`,onClick:()=>N("v2"),children:"V2 (Enhanced)"})]}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"8px"},children:"V2 features improved video quality, motion, and optional audio generation"})]}),I==="wan2.2"&&r.jsxs("div",{style:{backgroundColor:"var(--bg-tertiary)",padding:"16px",borderRadius:"8px",marginTop:"8px"},children:[r.jsxs("div",{onClick:()=>fs(!mn),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer"},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[r.jsx(ga,{size:16}),r.jsx("span",{style:{fontWeight:600,fontSize:"0.9rem"},children:"Workflow Presets"}),hn&&r.jsx("span",{style:{fontSize:"0.7rem",backgroundColor:"var(--accent-color)",color:"white",padding:"2px 6px",borderRadius:"4px",marginLeft:"4px"},children:hn.name})]}),r.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:mn?"▼":"▶"})]}),mn&&r.jsx("div",{style:{marginTop:"12px"},children:r.jsx(bx,{onPresetChange:E=>{var Z,de,pe,Be;if(ms(E),E!=null&&E.parameters){const tt=E.parameters;(Z=tt.steps)!=null&&Z.default&&B(tt.steps.default),(de=tt.cfg)!=null&&de.default&&te(tt.cfg.default),((pe=tt.seed)==null?void 0:pe.default)!==void 0&&T(tt.seed.default),(Be=tt.frame_rate)!=null&&Be.default&&O(tt.frame_rate.default)}},onParametersChange:E=>{hs(E),E.steps!==void 0&&B(E.steps),E.cfg!==void 0&&te(E.cfg),E.seed!==void 0&&T(E.seed),E.frame_rate!==void 0&&O(E.frame_rate)},currentParameters:{steps:M,cfg:Q,seed:oe,frame_rate:$}})})]}),I==="wan2.2"&&r.jsxs("div",{style:{backgroundColor:"var(--bg-tertiary)",padding:"16px",borderRadius:"8px",marginTop:"8px"},children:[r.jsxs("div",{onClick:()=>K(!v),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer"},children:[r.jsx("div",{style:{fontSize:"0.9rem",fontWeight:600,color:"var(--text-primary)"},children:"⚙️ Sampling Settings"}),r.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:v?"▼":"▶"})]}),v&&r.jsxs("div",{style:{marginTop:"12px"},children:[r.jsxs("div",{className:"form-group",style:{marginBottom:"12px"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"Sampling Steps"}),r.jsx("span",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:M})]}),r.jsx("input",{type:"range",min:"4",max:"20",step:"1",value:M,onChange:E=>B(parseInt(E.target.value,10)),style:{width:"100%",cursor:"pointer"}}),r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.7rem",color:"var(--text-muted)"},children:[r.jsx("span",{children:"4 (fast)"}),r.jsx("span",{children:"6 (rec)"}),r.jsx("span",{children:"20 (quality)"})]})]}),r.jsxs("div",{className:"form-group",style:{marginBottom:"12px"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"CFG Guidance"}),r.jsx("span",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:Q.toFixed(1)})]}),r.jsx("input",{type:"range",min:"1.0",max:"10.0",step:"0.5",value:Q,onChange:E=>te(parseFloat(E.target.value)),style:{width:"100%",cursor:"pointer"}}),r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.7rem",color:"var(--text-muted)"},children:[r.jsx("span",{children:"1.0 (rec)"}),r.jsx("span",{children:"5.0"}),r.jsx("span",{children:"10.0"})]})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Seed"}),r.jsxs("div",{style:{display:"flex",gap:"8px"},children:[r.jsx("input",{type:"number",value:oe,onChange:E=>T(parseInt(E.target.value,10)),placeholder:"-1 for random",style:{flex:1,padding:"8px 12px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"6px",color:"var(--text-primary)",fontSize:"0.9rem"}}),r.jsx("button",{className:"btn ghost sm",onClick:()=>T(-1),style:{whiteSpace:"nowrap"},children:"Random"})]}),r.jsx("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginTop:"4px"},children:"-1 = random seed each generation"})]})]}),r.jsxs("div",{style:{marginTop:"16px",paddingTop:"16px",borderTop:"1px solid var(--border-color)"},children:[r.jsxs("div",{onClick:()=>le(!F),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer",marginBottom:F?"12px":0},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[r.jsx(ni,{size:16}),r.jsx("span",{style:{fontWeight:600,fontSize:"0.9rem"},children:"LoRA Models"}),V.length>0&&r.jsxs("span",{style:{fontSize:"0.7rem",backgroundColor:"var(--accent-color)",color:"white",padding:"2px 6px",borderRadius:"4px"},children:[V.length," active"]})]}),r.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:F?"▼":"▶"})]}),F&&r.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"12px"},children:[V.map((E,Z)=>r.jsxs("div",{style:{backgroundColor:"var(--bg-input)",borderRadius:"8px",padding:"12px",border:"1px solid var(--border-color)"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"8px"},children:[r.jsxs("span",{style:{fontSize:"0.8rem",fontWeight:600},children:["LoRA #",Z+1]}),r.jsx("button",{onClick:()=>Y(V.filter((de,pe)=>pe!==Z)),style:{background:"transparent",border:"none",color:"#ef4444",cursor:"pointer",padding:"2px 6px",fontSize:"0.8rem"},children:"✕ Remove"})]}),r.jsxs("div",{style:{marginBottom:"8px"},children:[r.jsx("label",{style:{display:"block",fontSize:"0.75rem",color:"var(--text-muted)",marginBottom:"4px"},children:"High Noise (steps 0-3)"}),r.jsxs("select",{value:E.high||"",onChange:de=>{const pe=[...V];pe[Z]={...E,high:de.target.value},Y(pe)},style:{width:"100%",padding:"6px 10px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"4px",color:"var(--text-primary)",fontSize:"0.8rem"},children:[r.jsx("option",{value:"",children:"None"}),b.by_category&&Object.keys(b.by_category).sort().map(de=>r.jsx("optgroup",{label:de==="root"?"📁 Root":`📁 ${de}`,children:b.by_category[de].map(pe=>r.jsxs("option",{value:pe.path,children:[pe.name," (",pe.size_mb,"MB)"]},pe.path))},de))]})]}),r.jsxs("div",{style:{marginBottom:"8px"},children:[r.jsx("label",{style:{display:"block",fontSize:"0.75rem",color:"var(--text-muted)",marginBottom:"4px"},children:"Low Noise (steps 3+)"}),r.jsxs("select",{value:E.low||"",onChange:de=>{const pe=[...V];pe[Z]={...E,low:de.target.value},Y(pe)},style:{width:"100%",padding:"6px 10px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"4px",color:"var(--text-primary)",fontSize:"0.8rem"},children:[r.jsx("option",{value:"",children:"None (uses High Noise)"}),b.by_category&&Object.keys(b.by_category).sort().map(de=>r.jsx("optgroup",{label:de==="root"?"📁 Root":`📁 ${de}`,children:b.by_category[de].map(pe=>r.jsxs("option",{value:pe.path,children:[pe.name," (",pe.size_mb,"MB)"]},pe.path))},de))]})]}),r.jsxs("div",{children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"2px"},children:[r.jsx("label",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Strength"}),r.jsx("span",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:(E.strength||1).toFixed(2)})]}),r.jsx("input",{type:"range",min:"0",max:"2",step:"0.05",value:E.strength||1,onChange:de=>{const pe=[...V];pe[Z]={...E,strength:parseFloat(de.target.value)},Y(pe)},style:{width:"100%",cursor:"pointer"}})]})]},Z)),r.jsx("button",{onClick:()=>Y([...V,{high:"",low:"",strength:1}]),style:{padding:"8px 12px",backgroundColor:"transparent",border:"1px dashed var(--border-color)",borderRadius:"6px",color:"var(--text-secondary)",cursor:"pointer",fontSize:"0.85rem",display:"flex",alignItems:"center",justifyContent:"center",gap:"6px"},children:"+ Add LoRA"}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",fontStyle:"italic"},children:"💡 Stack multiple LoRAs for combined effects. Each LoRA has its own strength."})]})]})]}),r.jsxs("div",{className:"form-group",style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsxs("div",{children:[r.jsx("div",{className:"grok-section-label",style:{marginBottom:"4px"},children:"Generate Audio"}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Enable audio generation (increases credits)"})]}),r.jsxs("label",{className:"grok-switch",children:[r.jsx("input",{type:"checkbox"}),r.jsx("span",{className:"grok-slider"})]})]}),r.jsxs("div",{className:"form-group",style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsxs("div",{children:[r.jsx("div",{className:"grok-section-label",style:{marginBottom:"4px"},children:"Camera Fixed"}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Whether to fix the camera position"})]}),r.jsxs("label",{className:"grok-switch",children:[r.jsx("input",{type:"checkbox"}),r.jsx("span",{className:"grok-slider"})]})]}),r.jsxs("div",{className:"form-group",style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsxs("div",{children:[r.jsx("div",{className:"grok-section-label",style:{marginBottom:"4px"},children:"🎬 Extend Duration"}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Chain multiple clips sequentially"})]}),r.jsxs("label",{className:"grok-switch",children:[r.jsx("input",{type:"checkbox",checked:et,onChange:E=>{yr(E.target.checked),E.target.checked||xe(1)}}),r.jsx("span",{className:"grok-slider"})]})]}),et&&r.jsxs("div",{className:"form-group",style:{background:"linear-gradient(135deg, rgba(233, 69, 96, 0.1) 0%, rgba(233, 69, 96, 0.05) 100%)",borderRadius:"8px",padding:"12px",marginTop:"-8px",border:"1px solid rgba(233, 69, 96, 0.2)"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"8px"},children:[r.jsxs("div",{className:"grok-section-label",children:["Number of Clips: ",Ct]}),r.jsxs("div",{style:{fontSize:"0.75rem",color:"#e94560",background:"rgba(233, 69, 96, 0.15)",padding:"2px 8px",borderRadius:"10px",fontWeight:"600"},children:["≈ ",(h*Ct).toFixed(0),"s total"]})]}),r.jsx("input",{type:"range",min:"1",max:"5",value:Ct,onChange:E=>xe(parseInt(E.target.value)),style:{width:"100%",accentColor:"#e94560"}}),r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.7rem",color:"var(--text-muted)",marginTop:"4px"},children:[r.jsx("span",{children:"1"}),r.jsx("span",{children:"2"}),r.jsx("span",{children:"3"}),r.jsx("span",{children:"4"}),r.jsx("span",{children:"5"})]}),r.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"8px",fontStyle:"italic"},children:"🔗 Each clip continues from the last frame of the previous clip"})]})]}),r.jsxs("div",{className:"grok-card",children:[r.jsx("div",{className:"grok-card-header",children:r.jsx("div",{className:"grok-card-title",children:"Aspect Ratio"})}),r.jsx("div",{className:"aspect-grid",children:[{label:"Auto",icon:r.jsx(Ut,{size:16})},{label:"21:9",icon:r.jsx("div",{style:{width:"24px",height:"10px",border:"1px solid currentColor"}})},{label:"16:9",icon:r.jsx("div",{style:{width:"24px",height:"14px",border:"1px solid currentColor"}})},{label:"4:3",icon:r.jsx("div",{style:{width:"20px",height:"15px",border:"1px solid currentColor"}})},{label:"1:1",icon:r.jsx("div",{style:{width:"18px",height:"18px",border:"1px solid currentColor"}})},{label:"3:4",icon:r.jsx("div",{style:{width:"15px",height:"20px",border:"1px solid currentColor"}})},{label:"9:16",icon:r.jsx("div",{style:{width:"14px",height:"24px",border:"1px solid currentColor"}})}].map(E=>r.jsxs("button",{className:`aspect-btn ${X===E.label?"active":""}`,onClick:()=>A(E.label),children:[r.jsx("div",{className:"aspect-icon",style:{background:"transparent",display:"flex",alignItems:"center",justifyContent:"center",border:"none"},children:E.icon}),r.jsx("span",{className:"aspect-label",children:E.label})]},E.label))})]}),gn&&r.jsx("div",{style:{padding:"12px",backgroundColor:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.2)",borderRadius:"8px",color:"#ef4444",marginBottom:"16px",fontSize:"0.9rem"},children:gn}),r.jsx("button",{className:"primary-btn",disabled:!xs,onClick:J,style:{height:"48px",fontSize:"1rem",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px",backgroundColor:"#e5e5e5",color:"black"},children:jr?r.jsx(r.Fragment,{children:"Generating..."}):r.jsxs(r.Fragment,{children:[r.jsx(Ut,{size:18}),"Generate from Image"]})}),jr&&r.jsx("div",{className:"progress-container",children:r.jsx("div",{className:"progress-indeterminate"})})]})}const En={wan22:[{value:"wan2.2-t2i",label:"Wan2.2 T2I (Multi-GPU)",category:"Video Model"}],flux:[{value:"flux1-dev-fp8",label:"Flux.1 Dev (FP8)",category:"Flux"}],sdxl:[{value:"CyberRealistic_Pony_v14.1_FP16.safetensors",label:"CyberRealistic Pony",category:"Realistic/Pony"},{value:"dreamshaperXL_lightningDPMSDE.safetensors",label:"Dreamshaper Lightning",category:"General"},{value:"illustriousRealismBy_v10VAE.safetensors",label:"Illustrious Realism",category:"Realistic"},{value:"juggernautXL_ragnarok.safetensors",label:"Juggernaut XL",category:"General"},{value:"novaAnimeXL_ilV150.safetensors",label:"Nova Anime XL",category:"Anime"},{value:"ponyDiffusionV6XL_v6StartWithThisOne.safetensors",label:"Pony Diffusion V6",category:"Pony"},{value:"reapony_v90.safetensors",label:"Reapony V9",category:"Realistic/Pony"},{value:"ultraRealisticByStable_v20FP16.safetensors",label:"Ultra Realistic",category:"Realistic"},{value:"waiIllustriousSDXL_v160.safetensors",label:"Wai Illustrious",category:"Anime"}],sd15:[{value:"Realistic_Vision_V5.1.safetensors",label:"Realistic Vision V5.1",category:"Realistic"}],diffusers:[{value:"sd3.5-large-int8",label:"SD3.5 Large (INT8)"},{value:"realvisxl-v5.0",label:"RealVisXL V5.0"}]},Rt=e=>e==="wan2.2-t2i"?"wan22":e.startsWith("flux")?"flux":e==="Realistic_Vision_V5.1.safetensors"?"sd15":e.endsWith(".safetensors")?"sdxl":"diffusers";function Ex({onOutput:e}){const[t,n]=c.useState(""),[s,a]=c.useState("ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text"),[l,o]=c.useState("1:1"),[i,d]=c.useState("normal"),[u,y]=c.useState("CyberRealistic_Pony_v14.1_FP16.safetensors"),[g,x]=c.useState(1),[k,S]=c.useState(!1),[z,R]=c.useState(""),[f,p]=c.useState(0),[m,h]=c.useState(!1),[j,_]=c.useState([]),[P,I]=c.useState([{name:"None",strength:1},{name:"None",strength:1},{name:"None",strength:1}]),[G,H]=c.useState(30),[N,C]=c.useState(7.5),[L,X]=c.useState(3.5),[A,$]=c.useState(-1),[O,M]=c.useState("dpmpp_2m"),[B,Q]=c.useState("karras"),te=c.useRef(null);c.useEffect(()=>{(async()=>{try{const K=await fetch(`${ee}/loras`);if(K.ok){const b=await K.json();_(b.loras||[])}}catch(K){console.warn("Failed to fetch LoRAs:",K)}})()},[]);const oe=(v,K,b)=>{I(D=>{const V=[...D];return V[v]={...V[v],[K]:b},V})},T=async()=>{var K,b,D,V,Y;if(!t.trim())return;S(!0),p(0),R("");const v=async(F,le=120)=>{for(let re=0;re<le;re++){await new Promise(ie=>setTimeout(ie,1e3));try{const ie=await fetch(`${ee}/comfyui/job/${F}`);if(!ie.ok)continue;const me=await ie.json();if(me.status==="pending")p(Math.min(10,re));else if(me.status==="running")p(Math.min(90,10+re*2));else{if(me.status==="completed")return p(100),me;if(me.status==="failed")throw new Error("Generation failed")}}catch(ie){if(ie.message==="Generation failed")throw ie}}throw new Error("Generation timed out")};try{for(let F=0;F<g;F++){const le=`t2i-${Date.now()}-${Math.random().toString(36).slice(2,8)}`,re=new FormData;re.append("prompt",t),re.append("aspect_ratio",l);const ie=Rt(u);let me="/generate-image",Ze=!1;if(ie==="wan22")me="/generate-wan22-t2i",re.append("steps",G),re.append("seed",A),Ze=!0;else if(ie==="flux")me="/generate-flux",re.append("steps",G),re.append("guidance",L),re.append("seed",A),Ze=!0;else if(ie==="sdxl"){me="/generate-sdxl",re.append("checkpoint",u),re.append("negative_prompt",s),re.append("steps",G),re.append("cfg",N),re.append("seed",A),re.append("sampler_name",O),re.append("scheduler",B);const et=P.filter(yr=>yr.name&&yr.name!=="None");et.length>0&&re.append("lora_configs",JSON.stringify(et)),Ze=!0}else ie==="sd15"?(me="/generate-sd15",re.append("negative_prompt",s),re.append("steps",G),re.append("cfg",N),re.append("seed",A),re.append("sampler_name",O),re.append("scheduler",B),Ze=!0):(re.append("mode",i),re.append("model",u),re.append("job_id",le));const Re=await Nt(`${ee}${me}`,re);if(!Re.ok)throw new Error(((K=Re.data)==null?void 0:K.detail)||`Generation failed (status ${Re.status})`);console.log(`Batch ${F+1}/${g} queued:`,Re.data);let ht=(b=Re.data)==null?void 0:b.url,Xt=(D=Re.data)==null?void 0:D.filename;if(Ze&&((V=Re.data)!=null&&V.prompt_id)){const et=await v(Re.data.prompt_id);ht=et.url||et.output_image,Xt=ht==null?void 0:ht.split("/").pop()}const Lr=ht?`${ee}${ht}`:"";p(100),e({kind:"image",url:Lr,backendUrl:Lr,filename:Xt,meta:(Y=Re.data)==null?void 0:Y.meta})}}catch(F){console.error("Generation error:",F),R(F.message||"Failed to generate image")}finally{te.current&&(clearInterval(te.current),te.current=null),S(!1),setTimeout(()=>p(0),500)}};return r.jsxs("div",{className:"tool-container",children:[r.jsx("div",{className:"grok-card",children:r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Mode"}),r.jsxs("div",{className:"form-select",style:{display:"flex",alignItems:"center",gap:"8px",cursor:"pointer"},children:[r.jsx(Ut,{size:16,className:"text-primary"}),r.jsx("span",{children:"Normal"})]}),r.jsxs("div",{className:"info-badge",children:[r.jsx("span",{style:{color:"#93c5fd"},children:"Standard Quality"}),r.jsx("div",{style:{marginTop:"4px",opacity:.8},children:"Fast and efficient image generation (1 credit per image)"})]})]})}),r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Enter Image Prompt"}),r.jsxs("div",{style:{display:"flex",gap:"4px"},children:[r.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px",fontSize:"10px"},children:"T"}),r.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px",fontSize:"10px"},children:"✨"})]})]}),r.jsx("div",{style:{position:"relative"},children:r.jsx("textarea",{className:"form-textarea",value:t,onChange:v=>n(v.target.value),rows:4,placeholder:"A attractive blonde woman with cup f, tattoos, looking at me defiantly.",style:{backgroundColor:"#0f0f0f",border:"none",resize:"none",paddingBottom:"24px"}})})]}),r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Model"}),r.jsx("span",{className:"nav-badge",style:{fontSize:"0.7rem"},children:Rt(u).toUpperCase()})]}),r.jsxs("div",{style:{marginBottom:"12px"},children:[r.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"⚡ Flux (Best Quality)"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:En.flux.map(v=>r.jsx("button",{className:`grok-toggle-btn ${u===v.value?"active":""}`,onClick:()=>y(v.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},children:v.label},v.value))})]}),r.jsxs("div",{style:{marginBottom:"12px"},children:[r.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"🎨 SDXL Checkpoints"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:En.sdxl.map(v=>r.jsx("button",{className:`grok-toggle-btn ${u===v.value?"active":""}`,onClick:()=>y(v.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},title:v.category,children:v.label},v.value))})]}),r.jsxs("div",{style:{marginBottom:"12px"},children:[r.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"🖼️ SD 1.5 (Fast, Low VRAM)"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:En.sd15.map(v=>r.jsx("button",{className:`grok-toggle-btn ${u===v.value?"active":""}`,onClick:()=>y(v.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},children:v.label},v.value))})]}),r.jsxs("div",{style:{marginBottom:"12px"},children:[r.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"🎬 Wan2.2 (Video Model T2I)"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:En.wan22.map(v=>r.jsx("button",{className:`grok-toggle-btn ${u===v.value?"active":""}`,onClick:()=>y(v.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},children:v.label},v.value))})]}),r.jsxs("div",{children:[r.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"🐍 Diffusers (Python)"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:En.diffusers.map(v=>r.jsx("button",{className:`grok-toggle-btn ${u===v.value?"active":""}`,onClick:()=>y(v.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},children:v.label},v.value))})]})]}),(Rt(u)==="sdxl"||Rt(u)==="sd15")&&r.jsxs("div",{className:"grok-card",children:[r.jsx("div",{className:"grok-card-header",children:r.jsx("div",{className:"grok-card-title",children:"Negative Prompt"})}),r.jsx("textarea",{className:"form-textarea",value:s,onChange:v=>a(v.target.value),rows:2,placeholder:"ugly, deformed, blurry...",style:{backgroundColor:"#0f0f0f",border:"none",resize:"none",fontSize:"0.85rem"}})]}),r.jsxs("div",{className:"grok-card",children:[r.jsx("div",{className:"grok-card-header",children:r.jsx("div",{className:"grok-card-title",children:"Aspect Ratio"})}),r.jsx("div",{className:"aspect-grid",style:{gridTemplateColumns:"repeat(5, 1fr)"},children:[{label:"1:1",icon:r.jsx("div",{style:{width:"18px",height:"18px",border:"1px solid currentColor"}})},{label:"16:9",icon:r.jsx("div",{style:{width:"24px",height:"14px",border:"1px solid currentColor"}})},{label:"9:16",icon:r.jsx("div",{style:{width:"14px",height:"24px",border:"1px solid currentColor"}})},{label:"4:3",icon:r.jsx("div",{style:{width:"20px",height:"15px",border:"1px solid currentColor"}})},{label:"3:4",icon:r.jsx("div",{style:{width:"15px",height:"20px",border:"1px solid currentColor"}})},{label:"2:3",icon:r.jsx("div",{style:{width:"16px",height:"24px",border:"1px solid currentColor"}})},{label:"3:2",icon:r.jsx("div",{style:{width:"24px",height:"16px",border:"1px solid currentColor"}})},{label:"4:5",icon:r.jsx("div",{style:{width:"16px",height:"20px",border:"1px solid currentColor"}})},{label:"5:4",icon:r.jsx("div",{style:{width:"20px",height:"16px",border:"1px solid currentColor"}})},{label:"9:21",icon:r.jsx("div",{style:{width:"10px",height:"24px",border:"1px solid currentColor"}})},{label:"21:9",icon:r.jsx("div",{style:{width:"24px",height:"10px",border:"1px solid currentColor"}})}].map(v=>r.jsxs("button",{className:`aspect-btn ${l===v.label?"active":""}`,onClick:()=>o(v.label),style:{height:"60px"},children:[r.jsx("div",{className:"aspect-icon",style:{background:"transparent",display:"flex",alignItems:"center",justifyContent:"center",border:"none",marginBottom:"4px"},children:v.icon}),r.jsx("span",{className:"aspect-label",style:{fontSize:"0.65rem"},children:v.label})]},v.label))})]}),r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",style:{cursor:"pointer"},onClick:()=>h(!m),children:[r.jsx("div",{className:"grok-card-title",children:"Advanced Settings"}),r.jsx(Mt,{size:16,className:"text-muted",style:{transform:m?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.2s"}})]}),m&&r.jsxs(r.Fragment,{children:[r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Batch Count"}),r.jsx("div",{className:"grok-toggle-group",children:[1,2,3,4].map(v=>r.jsx("button",{className:`grok-toggle-btn ${g===v?"active":""}`,onClick:()=>x(v),children:v},v))})]}),Rt(u)==="flux"&&r.jsxs(r.Fragment,{children:[r.jsxs("div",{className:"form-group",style:{marginTop:"12px"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"Steps"}),r.jsx("span",{className:"nav-badge",children:G})]}),r.jsx("input",{type:"range",min:"10",max:"30",value:G,onChange:v=>H(parseInt(v.target.value)),className:"form-range"})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"Guidance"}),r.jsx("span",{className:"nav-badge",children:L})]}),r.jsx("input",{type:"range",min:"1",max:"10",step:"0.5",value:L,onChange:v=>X(parseFloat(v.target.value)),className:"form-range"})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Seed (-1 = random)"}),r.jsx("input",{type:"number",value:A,onChange:v=>$(parseInt(v.target.value)||-1),className:"form-input",style:{backgroundColor:"#0f0f0f",border:"1px solid #333",borderRadius:"6px",padding:"8px",width:"100%"}})]})]}),Rt(u)==="wan22"&&r.jsxs(r.Fragment,{children:[r.jsxs("div",{className:"form-group",style:{marginTop:"12px"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"Steps"}),r.jsx("span",{className:"nav-badge",children:G})]}),r.jsx("input",{type:"range",min:"10",max:"50",value:G,onChange:v=>H(parseInt(v.target.value)),className:"form-range"}),r.jsx("div",{style:{fontSize:"0.7rem",opacity:.6,marginTop:"4px"},children:"Multi-GPU workflow (DisTorch2) - 2-stage denoising"})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Seed (-1 = random)"}),r.jsx("input",{type:"number",value:A,onChange:v=>$(parseInt(v.target.value)||-1),className:"form-input",style:{backgroundColor:"#0f0f0f",border:"1px solid #333",borderRadius:"6px",padding:"8px",width:"100%"}})]})]}),(Rt(u)==="sdxl"||Rt(u)==="sd15")&&r.jsxs(r.Fragment,{children:[r.jsxs("div",{className:"form-group",style:{marginTop:"12px"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"Steps"}),r.jsx("span",{className:"nav-badge",children:G})]}),r.jsx("input",{type:"range",min:"10",max:"50",value:G,onChange:v=>H(parseInt(v.target.value)),className:"form-range"})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[r.jsx("label",{className:"grok-section-label",children:"CFG Scale"}),r.jsx("span",{className:"nav-badge",children:N})]}),r.jsx("input",{type:"range",min:"1",max:"15",step:"0.5",value:N,onChange:v=>C(parseFloat(v.target.value)),className:"form-range"})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Sampler"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"4px"},children:["euler","euler_ancestral","dpmpp_2m","dpmpp_sde"].map(v=>r.jsx("button",{className:`grok-toggle-btn ${O===v?"active":""}`,onClick:()=>M(v),style:{fontSize:"0.7rem",padding:"4px 8px"},children:v},v))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Scheduler"}),r.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"4px"},children:["normal","karras","exponential","sgm_uniform"].map(v=>r.jsx("button",{className:`grok-toggle-btn ${B===v?"active":""}`,onClick:()=>Q(v),style:{fontSize:"0.7rem",padding:"4px 8px"},children:v},v))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Seed (-1 = random)"}),r.jsx("input",{type:"number",value:A,onChange:v=>$(parseInt(v.target.value)||-1),className:"form-input",style:{backgroundColor:"#0f0f0f",border:"1px solid #333",borderRadius:"6px",padding:"8px",width:"100%"}})]}),Rt(u)==="sdxl"&&j.length>0&&r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",style:{marginBottom:"8px"},children:"LoRAs (up to 3)"}),P.map((v,K)=>r.jsxs("div",{style:{display:"flex",gap:"8px",marginBottom:"8px",alignItems:"center"},children:[r.jsxs("select",{value:v.name,onChange:b=>oe(K,"name",b.target.value),style:{flex:1,backgroundColor:"#0f0f0f",border:"1px solid #333",borderRadius:"6px",padding:"6px 8px",color:"#fff",fontSize:"0.75rem"},children:[r.jsx("option",{value:"None",children:"None"}),j.map(b=>r.jsx("option",{value:b,children:b.replace(".safetensors","")},b))]}),r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"4px",minWidth:"80px"},children:[r.jsx("input",{type:"range",min:"0",max:"2",step:"0.1",value:v.strength,onChange:b=>oe(K,"strength",parseFloat(b.target.value)),disabled:v.name==="None",style:{width:"50px"}}),r.jsx("span",{style:{fontSize:"0.7rem",opacity:v.name==="None"?.3:1},children:v.strength.toFixed(1)})]})]},K)),r.jsx("div",{style:{fontSize:"0.65rem",opacity:.5,marginTop:"4px"},children:"Strength: 0.5-1.0 recommended"})]})]})]})]}),z&&r.jsx("div",{style:{color:"#ef4444",marginBottom:"12px",fontSize:"0.9rem"},children:z}),r.jsx("button",{className:"primary-btn",onClick:T,disabled:k||!t.trim(),style:{height:"48px",fontSize:"1rem",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px",backgroundColor:"white",color:"black"},children:k?r.jsx(r.Fragment,{children:"Generating..."}):r.jsxs(r.Fragment,{children:[r.jsx(Ut,{size:18}),"Generate ",g>1?`${g} Images`:"Image"," (",g,")"]})}),k&&r.jsx("div",{className:"progress-container",children:r.jsx("div",{className:"progress-fill",style:{width:`${Math.min(f,100)}%`}})})]})}function zx({onOutput:e}){const[t,n]=c.useState(""),[s,a]=c.useState("16:9"),[l,o]=c.useState(!1),[i,d]=c.useState(null),[u,y]=c.useState(""),[g,x]=c.useState(16),[k,S]=c.useState(!1),z=async()=>{t.trim()&&(o(!0),setTimeout(()=>{o(!1),alert("Text-to-Image backend is not yet connected.")},1500))},R=async()=>{i&&(S(!0),setTimeout(()=>S(!1),2e3))};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Step 1: Text to Image"}),r.jsx(mr,{size:16,className:"text-muted"})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Image Prompt"}),r.jsx("textarea",{className:"form-textarea",value:t,onChange:f=>n(f.target.value),placeholder:"Describe the image you want to generate...",rows:3,style:{backgroundColor:"#0f0f0f",border:"none",resize:"none"}})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Aspect Ratio"}),r.jsx("div",{className:"aspect-grid",children:[{label:"16:9",icon:r.jsx("div",{style:{width:"24px",height:"14px",border:"2px solid currentColor",borderRadius:"2px"}})},{label:"9:16",icon:r.jsx("div",{style:{width:"14px",height:"24px",border:"2px solid currentColor",borderRadius:"2px"}})},{label:"1:1",icon:r.jsx("div",{style:{width:"20px",height:"20px",border:"2px solid currentColor",borderRadius:"2px"}})},{label:"21:9",icon:r.jsx("div",{style:{width:"28px",height:"12px",border:"2px solid currentColor",borderRadius:"2px"}})}].map(f=>r.jsxs("button",{className:`aspect-btn ${s===f.label?"active":""}`,onClick:()=>a(f.label),children:[r.jsx("div",{className:"aspect-icon",style:{background:"transparent",display:"flex",alignItems:"center",justifyContent:"center"},children:f.icon}),r.jsx("span",{className:"aspect-label",children:f.label})]},f.label))})]}),r.jsx("button",{className:"primary-btn",onClick:z,disabled:l||!t.trim(),style:{display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:l?"Generating Image...":r.jsxs(r.Fragment,{children:[r.jsx(Ut,{size:16})," Generate Image"]})})]}),r.jsxs("div",{className:`grok-card ${i?"":"opacity-50"}`,style:{transition:"opacity 0.3s"},children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Step 2: Animate"}),r.jsx(ls,{size:16,className:"text-muted"})]}),i?r.jsx("div",{className:"form-group",children:r.jsx("img",{src:i,alt:"Generated",style:{width:"100%",borderRadius:"8px",border:"1px solid var(--border-color)",marginBottom:"16px"}})}):r.jsx("div",{className:"upload-box",style:{padding:"24px",marginBottom:"16px",borderStyle:"dashed"},children:r.jsx("div",{className:"text-muted",children:"Generate an image above to continue"})}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Motion Prompt (Optional)"}),r.jsx("textarea",{className:"form-textarea",value:u,onChange:f=>y(f.target.value),placeholder:"Describe how the image should move...",rows:2,disabled:!i,style:{backgroundColor:"#0f0f0f",border:"none",resize:"none"}})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("label",{className:"grok-section-label",children:["Duration (",g," frames)"]}),r.jsx("input",{type:"range",min:"8",max:"32",step:"4",value:g,onChange:f=>x(parseInt(f.target.value,10)),disabled:!i,style:{width:"100%",accentColor:"var(--text-primary)"}})]}),r.jsx("button",{className:"primary-btn",onClick:R,disabled:!i||k,style:{display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:k?"Generating Video...":r.jsxs(r.Fragment,{children:[r.jsx(ls,{size:16})," Generate Video"]})})]})]})}const Tx=[{value:"none",label:"Custom",desc:"Use your own prompt"},{value:"anime",label:"Anime",desc:"Japanese animation style"},{value:"cartoon",label:"Cartoon",desc:"Cartoon/comic style"},{value:"sketch",label:"Sketch",desc:"Pencil sketch effect"},{value:"oil-painting",label:"Oil Painting",desc:"Classic oil painting style"},{value:"watercolor",label:"Watercolor",desc:"Watercolor painting effect"},{value:"pixel-art",label:"Pixel Art",desc:"Retro pixel art style"},{value:"cyberpunk",label:"Cyberpunk",desc:"Neon futuristic style"},{value:"3d-render",label:"3D Render",desc:"Modern 3D rendered look"}],Px={anime:"anime style, japanese animation, cel shading, vibrant colors, detailed linework",cartoon:"cartoon style, comic art, bold outlines, bright colors, disney style",sketch:"pencil sketch, hand-drawn, graphite, detailed linework, black and white","oil-painting":"oil painting style, classical art, brush strokes, rich colors, masterpiece",watercolor:"watercolor painting, soft edges, translucent colors, artistic, flowing","pixel-art":"pixel art style, 8-bit, retro gaming, blocky, nostalgic",cyberpunk:"cyberpunk style, neon lights, futuristic, rain, dark atmosphere, high tech","3d-render":"3d render, modern cgi, photorealistic, octane render, unreal engine"};function Ix({onOutput:e}){const[t,n]=c.useState(null),[s,a]=c.useState(null),[l,o]=c.useState(null),[i,d]=c.useState("none"),[u,y]=c.useState(""),[g,x]=c.useState("blurry, low quality, distorted, watermark"),[k,S]=c.useState(.5),[z,R]=c.useState(8),[f,p]=c.useState(32),[m,h]=c.useState(!1),[j,_]=c.useState(20),[P,I]=c.useState(7.5),[G,H]=c.useState(-1),[N,C]=c.useState(!1),[L,X]=c.useState(null),[A,$]=c.useState(""),[O,M]=c.useState(0),[B,Q]=c.useState(null),te=c.useRef(null),oe=c.useCallback(b=>{var V;const D=(V=b.target.files)==null?void 0:V[0];if(D){n(D);const Y=URL.createObjectURL(D);a(Y),Q(null),X(null);const F=document.createElement("video");F.onloadedmetadata=()=>{o({duration:F.duration.toFixed(1),width:F.videoWidth,height:F.videoHeight})},F.src=Y}},[]),T=c.useCallback(b=>{var V;b.preventDefault();const D=(V=b.dataTransfer.files)==null?void 0:V[0];if(D&&D.type.startsWith("video/")){n(D);const Y=URL.createObjectURL(D);a(Y),Q(null),X(null);const F=document.createElement("video");F.onloadedmetadata=()=>{o({duration:F.duration.toFixed(1),width:F.videoWidth,height:F.videoHeight})},F.src=Y}},[]),v=async(b,D=300)=>{for(let V=0;V<D;V++){await new Promise(Y=>setTimeout(Y,2e3));try{const Y=await fetch(`${ee}/comfyui/job/${b}`);if(!Y.ok)continue;const F=await Y.json();if(F.status==="pending")$("Queued..."),M(Math.min(5,V));else if(F.status==="running")$("Transforming video..."),M(Math.min(95,5+V*.5));else{if(F.status==="completed")return M(100),$("Done!"),F;if(F.status==="failed")throw new Error(F.error||"V2V failed")}}catch(Y){if(Y.message.includes("failed"))throw Y}}throw new Error("V2V timed out - video processing can take several minutes")},K=async()=>{var D,V,Y;if(!t)return;const b=i!=="none"?Px[i]+(u?", "+u:""):u;if(!b.trim()){X("Please select a style or enter a prompt");return}C(!0),X(null),$("Uploading..."),M(0);try{const F=new FormData;F.append("file",t),F.append("prompt",b),F.append("negative_prompt",g),F.append("denoise",String(k)),F.append("fps",String(z)),F.append("max_frames",String(f)),F.append("steps",String(j)),F.append("cfg",String(P)),F.append("seed",String(G));const le=await Nt(`${ee}/generate-v2v`,F);if(!le.ok)throw new Error(((D=le.data)==null?void 0:D.detail)||"V2V transform failed");const re=(V=le.data)==null?void 0:V.prompt_id;if(!re)throw new Error("No prompt_id returned");$("Queued...");const ie=await v(re);if(ie.output_video||ie.url){const me=ie.output_video||ie.url,Ze=me.startsWith("http")?me:`${ee}${me}`;Q(Ze),e&&e({kind:"video",url:Ze,filename:me.split("/").pop(),meta:(Y=le.data)==null?void 0:Y.meta})}}catch(F){console.error("V2V error:",F),X(F.message)}finally{C(!1),$(""),M(0)}};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(os,{size:18}),"Source Video"]}),r.jsxs("div",{className:`upload-dropzone ${s?"has-preview":""}`,onDrop:T,onDragOver:b=>b.preventDefault(),onClick:()=>document.getElementById("v2v-file-input").click(),children:[s?r.jsx("video",{ref:te,src:s,className:"upload-preview",controls:!0,muted:!0,loop:!0,style:{maxHeight:"250px"}}):r.jsxs("div",{className:"upload-placeholder",children:[r.jsx(St,{size:32}),r.jsx("p",{children:"Drop video here or click to upload"}),r.jsx("span",{style:{fontSize:"12px",opacity:.6},children:"MP4, WebM, MOV"})]}),r.jsx("input",{id:"v2v-file-input",type:"file",accept:"video/*",onChange:oe,style:{display:"none"}})]}),l&&r.jsxs("div",{className:"video-info",children:[r.jsxs("span",{children:["📐 ",l.width," × ",l.height,"px"]}),r.jsxs("span",{children:["⏱️ ",l.duration,"s"]})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(Ht,{size:18}),"Style Transform"]}),r.jsx("div",{className:"style-grid",children:Tx.map(b=>r.jsxs("button",{className:`style-btn ${i===b.value?"active":""}`,onClick:()=>d(b.value),children:[r.jsx("span",{className:"style-name",children:b.label}),r.jsx("span",{className:"style-desc",children:b.desc})]},b.value))})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:["Prompt ",i!=="none"&&r.jsx("span",{className:"hint",children:"(optional - adds to style)"})]}),r.jsx("textarea",{value:u,onChange:b=>y(b.target.value),placeholder:i!=="none"?"Add extra details to the style...":"Describe the desired look...",rows:3,className:"prompt-textarea"})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Transform Strength"}),r.jsxs("div",{className:"slider-row",children:[r.jsx("input",{type:"range",min:"0.1",max:"1",step:"0.05",value:k,onChange:b=>S(parseFloat(b.target.value))}),r.jsxs("span",{className:"slider-value",children:[(k*100).toFixed(0),"%"]})]}),r.jsxs("div",{className:"slider-labels",children:[r.jsx("span",{children:"Subtle"}),r.jsx("span",{children:"Complete"})]})]}),r.jsxs("div",{className:"tool-section collapsible",children:[r.jsxs("h3",{onClick:()=>h(!m),style:{cursor:"pointer"},children:[r.jsx(hr,{size:16}),"Advanced Settings",r.jsx(Mt,{size:16,style:{marginLeft:"auto",transform:m?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.2s"}})]}),m&&r.jsxs("div",{className:"advanced-content",children:[r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Output FPS"}),r.jsxs("select",{value:z,onChange:b=>R(parseInt(b.target.value)),children:[r.jsx("option",{value:8,children:"8 fps"}),r.jsx("option",{value:12,children:"12 fps"}),r.jsx("option",{value:16,children:"16 fps"}),r.jsx("option",{value:24,children:"24 fps"})]})]}),r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Max Frames"}),r.jsxs("select",{value:f,onChange:b=>p(parseInt(b.target.value)),children:[r.jsx("option",{value:16,children:"16 frames (~2s @8fps)"}),r.jsx("option",{value:32,children:"32 frames (~4s @8fps)"}),r.jsx("option",{value:48,children:"48 frames (~6s @8fps)"}),r.jsx("option",{value:64,children:"64 frames (~8s @8fps)"})]})]}),r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Steps"}),r.jsx("input",{type:"number",min:10,max:50,value:j,onChange:b=>_(parseInt(b.target.value))})]}),r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"CFG Scale"}),r.jsx("input",{type:"number",min:1,max:15,step:.5,value:P,onChange:b=>I(parseFloat(b.target.value))})]}),r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Seed (-1 = random)"}),r.jsx("input",{type:"number",value:G,onChange:b=>H(parseInt(b.target.value)||-1)})]}),r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Negative Prompt"}),r.jsx("textarea",{value:g,onChange:b=>x(b.target.value),rows:2,style:{fontSize:"12px"}})]})]})]}),N&&r.jsxs("div",{className:"progress-section",children:[r.jsx("div",{className:"progress-bar",children:r.jsx("div",{className:"progress-fill",style:{width:`${O}%`}})}),r.jsxs("div",{className:"progress-status",children:[r.jsx(Oe,{size:16,className:"spin"}),A]})]}),L&&r.jsxs("div",{className:"error-message",children:["⚠️ ",L]}),r.jsx("button",{className:"btn-primary btn-large",onClick:K,disabled:!t||N,children:N?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Transforming..."]}):r.jsxs(r.Fragment,{children:[r.jsx(Ht,{size:18}),"Transform Video"]})}),B&&r.jsxs("div",{className:"result-section",children:[r.jsx("h3",{children:"Result"}),r.jsx("video",{src:B,controls:!0,className:"result-video"}),r.jsx("a",{href:B,download:!0,className:"btn-secondary",style:{marginTop:12},children:"Download Video"})]}),r.jsx("style",{children:`
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
      `})]})}const Mx=[{value:"brief",label:"Brief",desc:"Short 1-2 sentence description"},{value:"detailed",label:"Detailed",desc:"Comprehensive scene analysis"},{value:"prompt",label:"Prompt Style",desc:"Optimized for AI generation"},{value:"timeline",label:"Timeline",desc:"Frame-by-frame breakdown"}],Rx=[{value:"smolvlm",label:"SmolVLM",desc:"Fast, lightweight vision model"},{value:"cogvlm",label:"CogVLM",desc:"High quality, slower"},{value:"llava",label:"LLaVA",desc:"Balanced quality/speed"}];function Lx(){const[e,t]=c.useState(null),[n,s]=c.useState(null),[a,l]=c.useState(null),[o,i]=c.useState("smolvlm"),[d,u]=c.useState("detailed"),[y,g]=c.useState(1),[x,k]=c.useState(8),[S,z]=c.useState(!1),[R,f]=c.useState(!1),[p,m]=c.useState(null),[h,j]=c.useState(""),[_,P]=c.useState(null),[I,G]=c.useState(!1),H=c.useRef(null),N=c.useCallback(A=>{var O;const $=(O=A.target.files)==null?void 0:O[0];if($){t($);const M=URL.createObjectURL($);s(M),P(null),m(null);const B=document.createElement("video");B.onloadedmetadata=()=>{l({duration:B.duration.toFixed(1),width:B.videoWidth,height:B.videoHeight})},B.src=M}},[]),C=c.useCallback(A=>{var O;A.preventDefault();const $=(O=A.dataTransfer.files)==null?void 0:O[0];if($&&$.type.startsWith("video/")){t($);const M=URL.createObjectURL($);s(M),P(null),m(null);const B=document.createElement("video");B.onloadedmetadata=()=>{l({duration:B.duration.toFixed(1),width:B.videoWidth,height:B.videoHeight})},B.src=M}},[]),L=async()=>{var A;if(e){f(!0),m(null),j("Uploading video...");try{const $=new FormData;$.append("file",e),$.append("model",o),$.append("mode",d),$.append("frame_interval",String(y)),$.append("max_frames",String(x)),j("Analyzing video...");const O=await Nt(`${ee}/caption-video`,$);if(!O.ok)throw new Error(((A=O.data)==null?void 0:A.detail)||"Video analysis failed");P(O.data)}catch($){console.error("V2T error:",$),m($.message)}finally{f(!1),j("")}}},X=async A=>{await navigator.clipboard.writeText(A),G(!0),setTimeout(()=>G(!1),2e3)};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(os,{size:18}),"Source Video"]}),r.jsxs("div",{className:`upload-dropzone ${n?"has-preview":""}`,onDrop:C,onDragOver:A=>A.preventDefault(),onClick:()=>document.getElementById("v2t-file-input").click(),children:[n?r.jsx("video",{ref:H,src:n,className:"upload-preview",controls:!0,muted:!0,style:{maxHeight:"200px"}}):r.jsxs("div",{className:"upload-placeholder",children:[r.jsx(St,{size:32}),r.jsx("p",{children:"Drop video here or click to upload"}),r.jsx("span",{style:{fontSize:"12px",opacity:.6},children:"MP4, WebM, MOV"})]}),r.jsx("input",{id:"v2t-file-input",type:"file",accept:"video/*",onChange:N,style:{display:"none"}})]}),a&&r.jsxs("div",{className:"video-info",children:[r.jsxs("span",{children:["📐 ",a.width," × ",a.height]}),r.jsxs("span",{children:["⏱️ ",a.duration,"s"]})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(yc,{size:18}),"Analysis Model"]}),r.jsx("div",{className:"model-grid",children:Rx.map(A=>r.jsxs("button",{className:`model-btn ${o===A.value?"active":""}`,onClick:()=>i(A.value),children:[r.jsx("span",{className:"model-name",children:A.label}),r.jsx("span",{className:"model-desc",children:A.desc})]},A.value))})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Output Style"}),r.jsx("div",{className:"mode-grid",children:Mx.map(A=>r.jsxs("button",{className:`mode-btn ${d===A.value?"active":""}`,onClick:()=>u(A.value),children:[r.jsx("span",{className:"mode-name",children:A.label}),r.jsx("span",{className:"mode-desc",children:A.desc})]},A.value))})]}),r.jsxs("div",{className:"tool-section collapsible",children:[r.jsxs("h3",{onClick:()=>z(!S),style:{cursor:"pointer"},children:[r.jsx(hr,{size:16}),"Advanced",r.jsx(Mt,{size:16,style:{marginLeft:"auto",transform:S?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.2s"}})]}),S&&r.jsxs("div",{className:"advanced-content",children:[r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Frame Interval"}),r.jsxs("select",{value:y,onChange:A=>g(parseFloat(A.target.value)),children:[r.jsx("option",{value:.5,children:"Every 0.5s"}),r.jsx("option",{value:1,children:"Every 1s"}),r.jsx("option",{value:2,children:"Every 2s"}),r.jsx("option",{value:5,children:"Every 5s"})]})]}),r.jsxs("div",{className:"form-row",children:[r.jsx("label",{children:"Max Frames"}),r.jsxs("select",{value:x,onChange:A=>k(parseInt(A.target.value)),children:[r.jsx("option",{value:4,children:"4 frames"}),r.jsx("option",{value:8,children:"8 frames"}),r.jsx("option",{value:16,children:"16 frames"}),r.jsx("option",{value:32,children:"32 frames"})]})]})]})]}),p&&r.jsxs("div",{className:"error-message",children:["⚠️ ",p]}),r.jsx("button",{className:"btn-primary btn-large",onClick:L,disabled:!e||R,children:R?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),h]}):r.jsxs(r.Fragment,{children:[r.jsx(yc,{size:18}),"Analyze Video"]})}),_&&r.jsxs("div",{className:"result-section",children:[r.jsxs("div",{className:"result-header",children:[r.jsx("h3",{children:"Description"}),r.jsxs("button",{className:"copy-btn",onClick:()=>X(_.caption||_.description),children:[I?r.jsx(Bu,{size:16}):r.jsx(At,{size:16}),I?"Copied!":"Copy"]})]}),r.jsx("div",{className:"result-text",children:_.caption||_.description}),_.timeline&&_.timeline.length>0&&r.jsxs("div",{className:"timeline-section",children:[r.jsx("h4",{children:"Timeline"}),_.timeline.map((A,$)=>r.jsxs("div",{className:"timeline-item",children:[r.jsxs("span",{className:"timeline-time",children:[A.time,"s"]}),r.jsx("span",{className:"timeline-desc",children:A.description})]},$))]}),_.prompt&&r.jsxs("div",{className:"prompt-section",children:[r.jsxs("div",{className:"prompt-header",children:[r.jsx("h4",{children:"AI Generation Prompt"}),r.jsx("button",{className:"copy-btn small",onClick:()=>X(_.prompt),children:r.jsx(At,{size:14})})]}),r.jsx("div",{className:"prompt-text",children:_.prompt})]})]}),r.jsx("style",{children:`
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
      `})]})}function Fx(){var a;const[e,t]=c.useState([{id:1,name:"Text Generation",status:"completed",description:"Generate prompt from keywords"},{id:2,name:"Text to Image",status:"ready",description:"Create base image"},{id:3,name:"Image to Video",status:"pending",description:"Animate the image"},{id:4,name:"Upscale",status:"pending",description:"Enhance resolution"}]),[n,s]=c.useState(2);return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Production Pipeline"}),r.jsx(qu,{size:16,className:"text-muted"})]}),r.jsx("div",{style:{display:"flex",flexDirection:"column",gap:"16px"},children:e.map((l,o)=>r.jsxs("div",{className:`pipeline-step ${n===l.id?"active":""}`,style:{display:"flex",alignItems:"center",gap:"16px",padding:"16px",backgroundColor:n===l.id?"#1a1a1a":"transparent",borderRadius:"8px",border:n===l.id?"1px solid var(--border-color)":"1px solid transparent",opacity:l.status==="pending"?.5:1},children:[r.jsx("div",{style:{width:"32px",height:"32px",borderRadius:"50%",backgroundColor:l.status==="completed"?"#22c55e":n===l.id?"var(--text-primary)":"#333",color:l.status==="completed"||n===l.id?"var(--bg-root)":"var(--text-secondary)",display:"flex",alignItems:"center",justifyContent:"center",fontWeight:"bold",fontSize:"0.9rem"},children:l.status==="completed"?r.jsx(Km,{size:18}):l.id}),r.jsxs("div",{style:{flex:1},children:[r.jsx("div",{style:{fontWeight:600,color:"var(--text-primary)"},children:l.name}),r.jsx("div",{style:{fontSize:"0.85rem",color:"var(--text-secondary)"},children:l.description})]}),o<e.length-1&&r.jsx(Fm,{size:16,className:"text-muted",style:{opacity:.3}})]},l.id))})]}),r.jsxs("div",{className:"grok-card",children:[r.jsx("div",{className:"grok-card-header",children:r.jsxs("div",{className:"grok-card-title",children:["Step Configuration: ",(a=e.find(l=>l.id===n))==null?void 0:a.name]})}),r.jsx("div",{className:"placeholder-state",style:{padding:"20px 0"},children:r.jsx("div",{className:"text-muted",children:"Configuration options for this step would appear here."})})]}),r.jsxs("button",{className:"primary-btn",style:{display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:[r.jsx(si,{size:18}),"Run Pipeline"]})]})}function Dx({onOutput:e}){const t=c.useRef(null),[n,s]=c.useState([]),[a,l]=c.useState(""),[o,i]=c.useState(10),[d,u]=c.useState(1e-4),[y,g]=c.useState(!1),[x,k]=c.useState(""),S=c.useMemo(()=>n.length>0&&a.trim().length>0&&!y,[n,a,y]),z=p=>{const m=Array.from(p||[]);s(m),k("")},R=()=>{s([]),t.current&&(t.current.value="")},f=async()=>{var m;if(n.length===0){k("At least one image is required");return}if(!a.trim()){k("Model name is required");return}g(!0),k("");const p=new FormData;n.forEach(h=>p.append("files",h)),p.append("model_name",a.trim()),p.append("num_epochs",String(o)),p.append("learning_rate",String(d));try{const h=await Nt(`${ee}/train-lora`,p);if(!h.ok){k(((m=h.data)==null?void 0:m.detail)||`Training failed (status ${h.status})`);return}e({kind:"lora",...h.data})}catch(h){const j=(h==null?void 0:h.message)||"Failed to start LoRA training";k(j),await La({level:"error",message:"LoRA training failed",timestamp:new Date().toISOString(),meta:{message:j}})}finally{g(!1)}};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Training Dataset"}),r.jsx(ni,{size:16,className:"text-muted"})]}),r.jsx("input",{ref:t,type:"file",accept:"image/*",multiple:!0,onChange:p=>z(p.target.files),style:{display:"none"}}),n.length===0?r.jsxs("div",{className:"upload-box",onClick:()=>{var p;return(p=t.current)==null?void 0:p.click()},style:{cursor:"pointer"},children:[r.jsx(St,{size:32,className:"text-muted"}),r.jsx("div",{className:"text-muted",children:"Upload training images (5-20 recommended)"}),r.jsxs("button",{className:"upload-btn",children:[r.jsx(St,{size:16}),"Select Images"]})]}):r.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"12px"},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsxs("span",{style:{color:"var(--text-primary)",fontWeight:500},children:[n.length," images selected"]}),r.jsxs("button",{onClick:R,className:"upload-btn secondary",style:{padding:"4px 8px",fontSize:"0.8rem"},children:[r.jsx(It,{size:14})," Clear"]})]}),r.jsx("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fill, minmax(60px, 1fr))",gap:"8px",maxHeight:"200px",overflowY:"auto",padding:"8px",backgroundColor:"#0f0f0f",borderRadius:"8px",border:"1px solid var(--border-color)"},children:n.map((p,m)=>r.jsx("div",{style:{aspectRatio:"1/1",backgroundColor:"#222",borderRadius:"4px",overflow:"hidden",display:"flex",alignItems:"center",justifyContent:"center"},children:r.jsx("span",{style:{fontSize:"0.6rem",color:"#666"},children:"IMG"})},m))})]})]}),r.jsxs("div",{className:"grok-card",children:[r.jsxs("div",{className:"grok-card-header",children:[r.jsx("div",{className:"grok-card-title",children:"Configuration"}),r.jsx(hr,{size:16,className:"text-muted"})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Model Name"}),r.jsx("input",{className:"form-input",value:a,onChange:p=>l(p.target.value),placeholder:"e.g. my-style-v1",style:{backgroundColor:"#0f0f0f"}})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("label",{className:"grok-section-label",children:["Training Epochs (",o,")"]}),r.jsx("input",{type:"range",min:"5",max:"50",step:"5",value:o,onChange:p=>i(parseInt(p.target.value,10)),style:{width:"100%",accentColor:"var(--text-primary)"}}),r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"4px"},children:[r.jsx("span",{children:"Fast (5)"}),r.jsx("span",{children:"Quality (50)"})]})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{className:"grok-section-label",children:"Learning Rate"}),r.jsx("input",{className:"form-input",type:"number",step:"0.00001",value:d,onChange:p=>u(parseFloat(p.target.value||"0")),style:{backgroundColor:"#0f0f0f"}})]})]}),x&&r.jsx("div",{style:{padding:"12px",backgroundColor:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.2)",borderRadius:"8px",color:"#ef4444",marginBottom:"16px",fontSize:"0.9rem"},children:x}),r.jsx("button",{className:"primary-btn",disabled:!S,onClick:f,style:{height:"48px",fontSize:"1rem",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:y?r.jsx(r.Fragment,{children:"Training..."}):r.jsxs(r.Fragment,{children:[r.jsx(Ju,{size:18}),"Start Training"]})})]})}const Ox=[{id:"brief",label:"Brief",description:"1-line summary"},{id:"detailed",label:"Detailed",description:"Full paragraph"},{id:"tags",label:"Tags",description:"Comma-separated keywords"},{id:"structured",label:"Structured",description:"Subject, style, mood"}],Ax=[{id:"florence2",label:"Florence-2",description:"Fast & accurate (Microsoft)"},{id:"blip2",label:"BLIP-2",description:"Detailed descriptions"},{id:"cogvlm",label:"CogVLM",description:"High quality (slower)"}];function $x({onSendToPrompt:e}){const[t,n]=c.useState(null),[s,a]=c.useState(null),[l,o]=c.useState("florence2"),[i,d]=c.useState("detailed"),[u,y]=c.useState(""),[g,x]=c.useState(!1),[k,S]=c.useState(null),z=c.useCallback(h=>{var _;const j=(_=h.target.files)==null?void 0:_[0];j&&(n(j),a(URL.createObjectURL(j)),y(""),S(null))},[]),R=c.useCallback(h=>{var _;h.preventDefault();const j=(_=h.dataTransfer.files)==null?void 0:_[0];j&&j.type.startsWith("image/")&&(n(j),a(URL.createObjectURL(j)),y(""),S(null))},[]),f=async()=>{if(t){x(!0),S(null);try{const h=new FormData;h.append("file",t),h.append("model",l),h.append("mode",i);const j=await fetch(`${ee}/caption-image`,{method:"POST",body:h});if(!j.ok){const P=await j.json();throw new Error(P.detail||"Caption failed")}const _=await j.json();y(_.caption||"")}catch(h){console.error("Caption error:",h),S(h.message)}finally{x(!1)}}},p=()=>{u&&navigator.clipboard.writeText(u)},m=()=>{u&&e&&e(u)};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(mr,{size:18}),"Upload Image"]}),r.jsxs("div",{className:`upload-dropzone ${s?"has-preview":""}`,onDrop:R,onDragOver:h=>h.preventDefault(),onClick:()=>document.getElementById("i2t-file-input").click(),children:[s?r.jsx("img",{src:s,alt:"Preview",className:"upload-preview"}):r.jsxs("div",{className:"upload-placeholder",children:[r.jsx(St,{size:32}),r.jsx("p",{children:"Drop image here or click to upload"})]}),r.jsx("input",{id:"i2t-file-input",type:"file",accept:"image/*",onChange:z,style:{display:"none"}})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(Ht,{size:18}),"Caption Settings"]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Model"}),r.jsx("select",{value:l,onChange:h=>o(h.target.value),children:Ax.map(h=>r.jsxs("option",{value:h.id,children:[h.label," - ",h.description]},h.id))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Caption Mode"}),r.jsx("div",{className:"button-group",children:Ox.map(h=>r.jsx("button",{className:`btn-option ${i===h.id?"active":""}`,onClick:()=>d(h.id),title:h.description,children:h.label},h.id))})]})]}),r.jsx("button",{className:"btn-primary btn-large",onClick:f,disabled:!t||g,children:g?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Generating caption..."]}):r.jsxs(r.Fragment,{children:[r.jsx(Ht,{size:18}),"Generate Caption"]})}),k&&r.jsxs("div",{className:"error-message",children:["⚠️ ",k]}),u&&r.jsxs("div",{className:"tool-section result-section",children:[r.jsx("h3",{children:"Generated Caption"}),r.jsxs("div",{className:"caption-result",children:[r.jsx("textarea",{value:u,onChange:h=>y(h.target.value),rows:4}),r.jsxs("div",{className:"caption-actions",children:[r.jsxs("button",{className:"btn-secondary",onClick:p,children:[r.jsx(At,{size:16}),"Copy"]}),e&&r.jsxs("button",{className:"btn-primary",onClick:m,children:[r.jsx(Yu,{size:16}),"Use as Prompt"]})]})]})]}),r.jsx("style",{children:`
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
      `})]})}const kc=[{id:"cinematic",label:"🎬 Cinematic",keywords:"cinematic lighting, film grain, dramatic shadows, professional photography"},{id:"anime",label:"🎌 Anime",keywords:"anime style, vibrant colors, cel shading, Japanese animation"},{id:"photorealistic",label:"📸 Photorealistic",keywords:"photorealistic, highly detailed, 8k, sharp focus, professional photo"},{id:"abstract",label:"🎨 Abstract",keywords:"abstract art, geometric shapes, vibrant colors, artistic"},{id:"vintage",label:"📼 Vintage",keywords:"vintage aesthetic, retro, film photography, nostalgic, 1970s"},{id:"cyberpunk",label:"🤖 Cyberpunk",keywords:"cyberpunk, neon lights, futuristic, dystopian, high tech low life"},{id:"fantasy",label:"🧙 Fantasy",keywords:"fantasy art, magical, ethereal lighting, mystical, enchanted"},{id:"minimalist",label:"⬜ Minimalist",keywords:"minimalist, clean, simple, negative space, modern"},{id:"horror",label:"👻 Horror",keywords:"dark atmosphere, eerie, horror, unsettling, creepy"},{id:"scifi",label:"🚀 Sci-Fi",keywords:"science fiction, futuristic, space, advanced technology"}];function Ux({onSendToTool:e}){const[t,n]=c.useState(""),[s,a]=c.useState(""),[l,o]=c.useState("expand"),[i,d]=c.useState(!0),[u,y]=c.useState(!1),[g,x]=c.useState(null),[k,S]=c.useState(!1),[z,R]=c.useState(null),f=async()=>{if(t.trim()){S(!0),R(null);try{const h=await fetch(`${ee}/generate-prompt`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({input:t.trim(),style:s||null,mode:l,include_negative:i,include_motion:u})});if(!h.ok){const _=await h.json();throw new Error(_.detail||"Generation failed")}const j=await h.json();x(j)}catch(h){console.error("Prompt generation error:",h),R(h.message)}finally{S(!1)}}},p=()=>{if(!t.trim())return;const h=t.trim(),j=kc.find(H=>H.id===s),_=j?`, ${j.keywords}`:"",P=`${h}${_}, masterpiece, best quality, highly detailed`;x({prompt:P,negative_prompt:i?"ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text, cropped, worst quality":"",motion_prompt:u?"smooth camera motion, cinematic movement, fluid animation":"",variations:null})},m=h=>{navigator.clipboard.writeText(h)};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(Ut,{size:18}),"Input Idea"]}),r.jsx("textarea",{value:t,onChange:h=>n(h.target.value),placeholder:"Describe your image or video idea... (e.g., 'a cat wearing sunglasses')",rows:3,className:"prompt-input"})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Style Preset"}),r.jsx("div",{className:"style-grid",children:kc.map(h=>r.jsx("button",{className:`style-btn ${s===h.id?"active":""}`,onClick:()=>a(s===h.id?"":h.id),children:h.label},h.id))})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Options"}),r.jsxs("div",{className:"options-row",children:[r.jsxs("label",{className:"checkbox-label",children:[r.jsx("input",{type:"checkbox",checked:i,onChange:h=>d(h.target.checked)}),"Generate negative prompt"]}),r.jsxs("label",{className:"checkbox-label",children:[r.jsx("input",{type:"checkbox",checked:u,onChange:h=>y(h.target.checked)}),"Include motion prompts (for video)"]})]})]}),r.jsxs("div",{className:"button-row",children:[r.jsxs("button",{className:"btn-primary btn-large",onClick:p,disabled:!t.trim(),children:[r.jsx(Ht,{size:18}),"Quick Generate"]}),r.jsx("button",{className:"btn-secondary btn-large",onClick:f,disabled:!t.trim()||k,title:"Uses AI for smarter enhancement (requires LLM)",children:k?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Generating..."]}):r.jsxs(r.Fragment,{children:[r.jsx(Ut,{size:18}),"AI Enhance"]})})]}),z&&r.jsxs("div",{className:"error-message",children:["⚠️ ",z]}),g&&r.jsxs("div",{className:"results-section",children:[r.jsxs("div",{className:"result-card",children:[r.jsxs("div",{className:"result-header",children:[r.jsx("h4",{children:"✨ Enhanced Prompt"}),r.jsx("button",{className:"btn-icon",onClick:()=>m(g.prompt),children:r.jsx(At,{size:16})})]}),r.jsx("p",{className:"result-text",children:g.prompt})]}),g.negative_prompt&&r.jsxs("div",{className:"result-card",children:[r.jsxs("div",{className:"result-header",children:[r.jsx("h4",{children:"🚫 Negative Prompt"}),r.jsx("button",{className:"btn-icon",onClick:()=>m(g.negative_prompt),children:r.jsx(At,{size:16})})]}),r.jsx("p",{className:"result-text muted",children:g.negative_prompt})]}),g.motion_prompt&&r.jsxs("div",{className:"result-card",children:[r.jsxs("div",{className:"result-header",children:[r.jsx("h4",{children:"🎬 Motion Prompt"}),r.jsx("button",{className:"btn-icon",onClick:()=>m(g.motion_prompt),children:r.jsx(At,{size:16})})]}),r.jsx("p",{className:"result-text",children:g.motion_prompt})]}),g.variations&&g.variations.length>0&&r.jsxs("div",{className:"result-card",children:[r.jsx("h4",{children:"🔄 Variations"}),g.variations.map((h,j)=>r.jsxs("div",{className:"variation-item",children:[r.jsx("p",{className:"result-text",children:h}),r.jsx("button",{className:"btn-icon",onClick:()=>m(h),children:r.jsx(At,{size:16})})]},j))]}),e&&r.jsxs("button",{className:"btn-primary",onClick:()=>e(g),children:[r.jsx(Yu,{size:16}),"Send to Generator"]})]}),r.jsx("style",{children:`
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
      `})]})}const Vx=[{value:"CyberRealistic_Pony_v14.1_FP16.safetensors",label:"CyberRealistic Pony"},{value:"dreamshaperXL_lightningDPMSDE.safetensors",label:"Dreamshaper Lightning"},{value:"juggernautXL_ragnarok.safetensors",label:"Juggernaut XL"},{value:"waiIllustriousSDXL_v160.safetensors",label:"Wai Illustrious (Anime)"}];function Bx({onOutput:e}){const[t,n]=c.useState(null),[s,a]=c.useState(null),[l,o]=c.useState(""),[i,d]=c.useState("ugly, deformed, blurry, low quality, bad anatomy, watermark"),[u,y]=c.useState(.6),[g,x]=c.useState("CyberRealistic_Pony_v14.1_FP16.safetensors"),[k,S]=c.useState(!1),[z,R]=c.useState(25),[f,p]=c.useState(7),[m,h]=c.useState(-1),[j,_]=c.useState("dpmpp_2m"),[P,I]=c.useState("karras"),[G,H]=c.useState(!1),[N,C]=c.useState(null),[L,X]=c.useState(""),[A,$]=c.useState(0),[O,M]=c.useState(null),B=c.useCallback(T=>{var K;const v=(K=T.target.files)==null?void 0:K[0];v&&(n(v),a(URL.createObjectURL(v)),M(null),C(null))},[]),Q=c.useCallback(T=>{var K;T.preventDefault();const v=(K=T.dataTransfer.files)==null?void 0:K[0];v&&v.type.startsWith("image/")&&(n(v),a(URL.createObjectURL(v)),M(null),C(null))},[]),te=async(T,v=120)=>{for(let K=0;K<v;K++){await new Promise(b=>setTimeout(b,1e3));try{const b=await fetch(`${ee}/comfyui/job/${T}`);if(!b.ok)continue;const D=await b.json();if(D.status==="pending")X("Queued..."),$(Math.min(10,K));else if(D.status==="running")X("Processing..."),$(Math.min(90,10+K*2));else{if(D.status==="completed")return $(100),X("Done!"),D;if(D.status==="failed")throw new Error(D.error||"Generation failed")}}catch(b){if(b.message.includes("failed"))throw b}}throw new Error("Generation timed out")},oe=async()=>{var T,v,K;if(t){H(!0),C(null),X("Uploading..."),$(0);try{const b=new FormData;b.append("file",t),b.append("prompt",l||"high quality, detailed"),b.append("negative_prompt",i),b.append("denoise",String(u)),b.append("checkpoint",g),b.append("steps",String(z)),b.append("cfg",String(f)),b.append("seed",String(m)),b.append("sampler_name",j),b.append("scheduler",P);const D=await Nt(`${ee}/generate-i2i`,b);if(!D.ok)throw new Error(((T=D.data)==null?void 0:T.detail)||"Generation failed");const V=(v=D.data)==null?void 0:v.prompt_id;if(!V)throw new Error("No prompt_id returned");X("Queued...");const Y=await te(V);if(Y.output_image||Y.url){const F=Y.output_image||Y.url,le=F.startsWith("http")?F:`${ee}${F}`;M(le),e&&e({kind:"image",url:le,filename:F.split("/").pop(),meta:(K=D.data)==null?void 0:K.meta})}}catch(b){console.error("I2I error:",b),C(b.message)}finally{H(!1),X(""),$(0)}}};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(mr,{size:18}),"Source Image"]}),r.jsxs("div",{className:`upload-dropzone ${s?"has-preview":""}`,onDrop:Q,onDragOver:T=>T.preventDefault(),onClick:()=>document.getElementById("i2i-file-input").click(),children:[s?r.jsx("img",{src:s,alt:"Preview",className:"upload-preview"}):r.jsxs("div",{className:"upload-placeholder",children:[r.jsx(St,{size:32}),r.jsx("p",{children:"Drop image here or click to upload"})]}),r.jsx("input",{id:"i2i-file-input",type:"file",accept:"image/*",onChange:B,style:{display:"none"}})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(Ht,{size:18}),"Transformation"]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Prompt (describe desired changes)"}),r.jsx("textarea",{value:l,onChange:T=>o(T.target.value),rows:3,placeholder:"Describe what you want the image to become... (e.g., 'anime style illustration')"})]}),r.jsxs("div",{className:"form-group",children:[r.jsxs("label",{children:[r.jsx(ga,{size:14}),"Denoise Strength",r.jsx("span",{className:"label-value",children:u.toFixed(2)})]}),r.jsx("input",{type:"range",min:"0.1",max:"1.0",step:"0.05",value:u,onChange:T=>y(parseFloat(T.target.value))}),r.jsxs("div",{className:"range-labels",children:[r.jsx("span",{children:"Subtle (0.1)"}),r.jsx("span",{children:"Complete (1.0)"})]}),r.jsxs("div",{className:"denoise-hint",children:[u<.3&&"💡 Minor adjustments, preserves most of original",u>=.3&&u<.6&&"💡 Moderate changes, good balance",u>=.6&&u<.8&&"💡 Significant transformation",u>=.8&&"💡 Near-complete regeneration from prompt"]})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Model"}),r.jsx("select",{value:g,onChange:T=>x(T.target.value),children:Vx.map(T=>r.jsx("option",{value:T.value,children:T.label},T.value))})]})]}),r.jsxs("div",{className:"tool-section collapsible",children:[r.jsxs("button",{className:"section-toggle",onClick:()=>S(!k),children:[r.jsx(hr,{size:16}),"Advanced Settings",r.jsx(Mt,{size:16,className:k?"rotated":""})]}),k&&r.jsxs("div",{className:"advanced-content",children:[r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Negative Prompt"}),r.jsx("textarea",{value:i,onChange:T=>d(T.target.value),rows:2})]}),r.jsxs("div",{className:"form-row",children:[r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"Steps"}),r.jsx("input",{type:"number",value:z,onChange:T=>R(parseInt(T.target.value)||25),min:"1",max:"50"})]}),r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"CFG Scale"}),r.jsx("input",{type:"number",value:f,onChange:T=>p(parseFloat(T.target.value)||7),min:"1",max:"20",step:"0.5"})]})]}),r.jsxs("div",{className:"form-row",children:[r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"Sampler"}),r.jsxs("select",{value:j,onChange:T=>_(T.target.value),children:[r.jsx("option",{value:"euler",children:"Euler"}),r.jsx("option",{value:"euler_ancestral",children:"Euler Ancestral"}),r.jsx("option",{value:"dpmpp_2m",children:"DPM++ 2M"}),r.jsx("option",{value:"dpmpp_2m_sde",children:"DPM++ 2M SDE"}),r.jsx("option",{value:"dpmpp_3m_sde",children:"DPM++ 3M SDE"})]})]}),r.jsxs("div",{className:"form-group half",children:[r.jsx("label",{children:"Scheduler"}),r.jsxs("select",{value:P,onChange:T=>I(T.target.value),children:[r.jsx("option",{value:"normal",children:"Normal"}),r.jsx("option",{value:"karras",children:"Karras"}),r.jsx("option",{value:"exponential",children:"Exponential"}),r.jsx("option",{value:"sgm_uniform",children:"SGM Uniform"})]})]})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Seed (-1 = random)"}),r.jsx("input",{type:"number",value:m,onChange:T=>h(parseInt(T.target.value)||-1)})]})]})]}),G&&r.jsxs("div",{className:"progress-section",children:[r.jsx("div",{className:"progress-bar",children:r.jsx("div",{className:"progress-fill",style:{width:`${A}%`}})}),r.jsxs("div",{className:"progress-status",children:[r.jsx(Oe,{size:16,className:"spin"}),L]})]}),N&&r.jsxs("div",{className:"error-message",children:["⚠️ ",N]}),r.jsx("button",{className:"btn-primary btn-large",onClick:oe,disabled:!t||G,children:G?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Transforming..."]}):r.jsxs(r.Fragment,{children:[r.jsx(Ht,{size:18}),"Transform Image"]})}),O&&r.jsxs("div",{className:"result-section",children:[r.jsx("h3",{children:"Result"}),r.jsxs("div",{className:"comparison",children:[r.jsxs("div",{className:"comparison-item",children:[r.jsx("span",{className:"comparison-label",children:"Original"}),r.jsx("img",{src:s,alt:"Original"})]}),r.jsxs("div",{className:"comparison-item",children:[r.jsx("span",{className:"comparison-label",children:"Transformed"}),r.jsx("img",{src:O,alt:"Result"})]})]})]}),r.jsx("style",{children:`
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
      `})]})}const Sc=[{value:"RealESRGAN_x4plus.pth",label:"RealESRGAN 4x (General)",scale:4},{value:"RealESRGAN_x4plus_anime_6B.pth",label:"RealESRGAN 4x (Anime)",scale:4},{value:"RealESRGAN_x2plus.pth",label:"RealESRGAN 2x",scale:2},{value:"4x-UltraSharp.pth",label:"4x UltraSharp",scale:4},{value:"4x_NMKD-Siax_200k.pth",label:"4x NMKD-Siax",scale:4}],Wx=[2,4];function Gx({onOutput:e}){const[t,n]=c.useState(null),[s,a]=c.useState(null),[l,o]=c.useState(null),[i,d]=c.useState("RealESRGAN_x4plus.pth"),[u,y]=c.useState(4),[g,x]=c.useState(!1),[k,S]=c.useState(!1),[z,R]=c.useState(null),[f,p]=c.useState(""),[m,h]=c.useState(0),[j,_]=c.useState(null),P=c.useCallback(L=>{var A;const X=(A=L.target.files)==null?void 0:A[0];if(X){n(X);const $=URL.createObjectURL(X);a($),_(null),R(null);const O=new Image;O.onload=()=>{o({width:O.width,height:O.height})},O.src=$}},[]),I=c.useCallback(L=>{var A;L.preventDefault();const X=(A=L.dataTransfer.files)==null?void 0:A[0];if(X&&X.type.startsWith("image/")){n(X);const $=URL.createObjectURL(X);a($),_(null),R(null);const O=new Image;O.onload=()=>{o({width:O.width,height:O.height})},O.src=$}},[]),G=async(L,X=120)=>{for(let A=0;A<X;A++){await new Promise($=>setTimeout($,1e3));try{const $=await fetch(`${ee}/comfyui/job/${L}`);if(!$.ok)continue;const O=await $.json();if(O.status==="pending")p("Queued..."),h(Math.min(10,A));else if(O.status==="running")p("Upscaling..."),h(Math.min(90,10+A*2));else{if(O.status==="completed")return h(100),p("Done!"),O;if(O.status==="failed")throw new Error(O.error||"Upscaling failed")}}catch($){if($.message.includes("failed"))throw $}}throw new Error("Upscaling timed out")},H=async()=>{var L,X,A;if(t){S(!0),R(null),p("Uploading..."),h(0);try{const $=new FormData;$.append("file",t),$.append("model",i),$.append("scale",String(u)),$.append("face_enhance",String(g));const O=await Nt(`${ee}/upscale`,$);if(!O.ok)throw new Error(((L=O.data)==null?void 0:L.detail)||"Upscaling failed");const M=(X=O.data)==null?void 0:X.prompt_id;if(!M)throw new Error("No prompt_id returned");p("Queued...");const B=await G(M);if(B.output_image||B.url){const Q=B.output_image||B.url,te=Q.startsWith("http")?Q:`${ee}${Q}`;_(te),e&&e({kind:"image",url:te,filename:Q.split("/").pop(),meta:(A=O.data)==null?void 0:A.meta})}}catch($){console.error("Upscale error:",$),R($.message)}finally{S(!1),p(""),h(0)}}};Sc.find(L=>L.value===i);const N=l?l.width*u:0,C=l?l.height*u:0;return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(mr,{size:18}),"Source Image"]}),r.jsxs("div",{className:`upload-dropzone ${s?"has-preview":""}`,onDrop:I,onDragOver:L=>L.preventDefault(),onClick:()=>document.getElementById("upscale-file-input").click(),children:[s?r.jsx("img",{src:s,alt:"Preview",className:"upload-preview"}):r.jsxs("div",{className:"upload-placeholder",children:[r.jsx(St,{size:32}),r.jsx("p",{children:"Drop image here or click to upload"})]}),r.jsx("input",{id:"upscale-file-input",type:"file",accept:"image/*",onChange:P,style:{display:"none"}})]}),l&&r.jsxs("div",{className:"image-info",children:[r.jsxs("span",{children:["📐 ",l.width," × ",l.height,"px"]}),r.jsx("span",{children:"→"}),r.jsxs("span",{className:"output-size",children:[N," × ",C,"px"]})]})]}),r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(bc,{size:18}),"Upscale Settings"]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Scale Factor"}),r.jsx("div",{className:"button-group",children:Wx.map(L=>r.jsxs("button",{className:`btn-option ${u===L?"active":""}`,onClick:()=>y(L),type:"button",children:[L,"x"]},L))})]}),r.jsxs("div",{className:"form-group",children:[r.jsx("label",{children:"Upscale Model"}),r.jsx("select",{value:i,onChange:L=>d(L.target.value),children:Sc.map(L=>r.jsx("option",{value:L.value,children:L.label},L.value))})]}),r.jsx("div",{className:"form-group",children:r.jsxs("label",{className:"checkbox-label",children:[r.jsx("input",{type:"checkbox",checked:g,onChange:L=>x(L.target.checked)}),"Face Enhancement (GFPGAN)",r.jsx("span",{className:"hint",children:"Improves face details"})]})})]}),k&&r.jsxs("div",{className:"progress-section",children:[r.jsx("div",{className:"progress-bar",children:r.jsx("div",{className:"progress-fill",style:{width:`${m}%`}})}),r.jsxs("div",{className:"progress-status",children:[r.jsx(Oe,{size:16,className:"spin"}),f]})]}),z&&r.jsxs("div",{className:"error-message",children:["⚠️ ",z]}),r.jsx("button",{className:"btn-primary btn-large",onClick:H,disabled:!t||k,children:k?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Upscaling..."]}):r.jsxs(r.Fragment,{children:[r.jsx(bc,{size:18}),"Upscale Image"]})}),j&&r.jsxs("div",{className:"result-section",children:[r.jsxs("h3",{children:["Result (",u,"x Upscaled)"]}),r.jsx("div",{className:"result-image",children:r.jsx("img",{src:j,alt:"Upscaled"})}),r.jsx("a",{href:j,download:!0,className:"btn-secondary",style:{marginTop:12,display:"inline-flex",alignItems:"center",gap:8},children:"Download Full Resolution"})]}),r.jsx("style",{children:`
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
      `})]})}const Hx=[{value:"alloy",label:"Alloy",desc:"Neutral, versatile"},{value:"echo",label:"Echo",desc:"Warm, conversational"},{value:"fable",label:"Fable",desc:"Expressive, dramatic"},{value:"onyx",label:"Onyx",desc:"Deep, authoritative"},{value:"nova",label:"Nova",desc:"Friendly, upbeat"},{value:"shimmer",label:"Shimmer",desc:"Soft, gentle"}],Qx=[{value:"tts",label:"Text to Speech",icon:r.jsx(Eh,{size:18}),desc:"Generate voice from text"},{value:"music",label:"Music Generation",icon:r.jsx(Rh,{size:18}),desc:"Generate music/sounds"},{value:"sfx",label:"Sound Effects",icon:r.jsx(oo,{size:18}),desc:"Generate sound effects"}],Xx=[{value:"ambient",label:"Ambient"},{value:"cinematic",label:"Cinematic"},{value:"electronic",label:"Electronic"},{value:"jazz",label:"Jazz"},{value:"classical",label:"Classical"},{value:"lofi",label:"Lo-Fi"},{value:"rock",label:"Rock"},{value:"hiphop",label:"Hip-Hop"}];function Yx({onOutput:e}){const[t,n]=c.useState("tts"),[s,a]=c.useState(""),[l,o]=c.useState("nova"),[i,d]=c.useState("cinematic"),[u,y]=c.useState(10),[g,x]=c.useState(!1),[k,S]=c.useState(1),[z,R]=c.useState(1),[f,p]=c.useState(!1),[m,h]=c.useState(null),[j,_]=c.useState(""),[P,I]=c.useState(0),[G,H]=c.useState(null),[N,C]=c.useState(!1),L=c.useRef(null),X=async(O,M=120)=>{for(let B=0;B<M;B++){await new Promise(Q=>setTimeout(Q,1e3));try{const Q=await fetch(`${ee}/comfyui/job/${O}`);if(!Q.ok)continue;const te=await Q.json();if(te.status==="pending")_("Queued..."),I(Math.min(10,B*2));else if(te.status==="running")_("Generating audio..."),I(Math.min(90,10+B*2));else{if(te.status==="completed")return I(100),te;if(te.status==="failed")throw new Error(te.error||"Generation failed")}}catch(Q){if(Q.message.includes("failed"))throw Q}}throw new Error("Generation timed out")},A=async()=>{var O,M,B;if(s.trim()){p(!0),h(null),_("Starting..."),I(0),H(null);try{let Q="/generate-audio";const te={text:s.trim(),mode:t};t==="tts"?(te.voice=l,te.speed=k,te.pitch=z):t==="music"?(te.style=i,te.duration=u):t==="sfx"&&(te.duration=Math.min(u,5));const oe=await xx(`${ee}${Q}`,te);if(!oe.ok)throw new Error(((O=oe.data)==null?void 0:O.detail)||"Audio generation failed");if((M=oe.data)!=null&&M.prompt_id){_("Queued...");const T=await X(oe.data.prompt_id);if(T.output_audio||T.url){const v=T.output_audio||T.url,K=v.startsWith("http")?v:`${ee}${v}`;H({url:K,filename:v.split("/").pop()})}}else if((B=oe.data)!=null&&B.url){const T=oe.data.url,v=T.startsWith("http")?T:`${ee}${T}`;H({url:v,filename:T.split("/").pop()})}e&&G&&e({kind:"audio",url:G.url,filename:G.filename})}catch(Q){console.error("Audio error:",Q),h(Q.message)}finally{p(!1),_(""),I(0)}}},$=()=>{L.current&&(N?L.current.pause():L.current.play(),C(!N))};return r.jsxs("div",{className:"tool-container",children:[r.jsxs("div",{className:"tool-section",children:[r.jsxs("h3",{children:[r.jsx(oo,{size:18}),"Generation Mode"]}),r.jsx("div",{className:"mode-grid",children:Qx.map(O=>r.jsxs("button",{className:`mode-btn ${t===O.value?"active":""}`,onClick:()=>n(O.value),children:[O.icon,r.jsx("span",{className:"mode-name",children:O.label}),r.jsx("span",{className:"mode-desc",children:O.desc})]},O.value))})]}),r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:t==="tts"?"Text to Speak":t==="music"?"Music Prompt":"Sound Description"}),r.jsx("textarea",{value:s,onChange:O=>a(O.target.value),placeholder:t==="tts"?"Enter the text you want to convert to speech...":t==="music"?'Describe the music you want to generate (e.g., "upbeat electronic dance track with heavy bass")':'Describe the sound effect (e.g., "thunder rumbling in the distance")',rows:4,className:"prompt-textarea"})]}),t==="tts"&&r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Voice"}),r.jsx("div",{className:"voice-grid",children:Hx.map(O=>r.jsxs("button",{className:`voice-btn ${l===O.value?"active":""}`,onClick:()=>o(O.value),children:[r.jsx("span",{className:"voice-name",children:O.label}),r.jsx("span",{className:"voice-desc",children:O.desc})]},O.value))})]}),t==="music"&&r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Style"}),r.jsx("div",{className:"style-grid",children:Xx.map(O=>r.jsx("button",{className:`style-btn ${i===O.value?"active":""}`,onClick:()=>d(O.value),children:O.label},O.value))})]}),(t==="music"||t==="sfx")&&r.jsxs("div",{className:"tool-section",children:[r.jsx("h3",{children:"Duration"}),r.jsxs("div",{className:"slider-row",children:[r.jsx("input",{type:"range",min:t==="sfx"?1:5,max:t==="sfx"?10:30,value:u,onChange:O=>y(parseInt(O.target.value))}),r.jsxs("span",{className:"slider-value",children:[u,"s"]})]})]}),t==="tts"&&r.jsxs("div",{className:"tool-section collapsible",children:[r.jsxs("h3",{onClick:()=>x(!g),style:{cursor:"pointer"},children:[r.jsx(hr,{size:16}),"Advanced",r.jsx(Mt,{size:16,style:{marginLeft:"auto",transform:g?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.2s"}})]}),g&&r.jsxs("div",{className:"advanced-content",children:[r.jsxs("div",{className:"slider-row",children:[r.jsx("label",{children:"Speed"}),r.jsx("input",{type:"range",min:.5,max:2,step:.1,value:k,onChange:O=>S(parseFloat(O.target.value))}),r.jsxs("span",{className:"slider-value",children:[k.toFixed(1),"x"]})]}),r.jsxs("div",{className:"slider-row",children:[r.jsx("label",{children:"Pitch"}),r.jsx("input",{type:"range",min:.5,max:2,step:.1,value:z,onChange:O=>R(parseFloat(O.target.value))}),r.jsxs("span",{className:"slider-value",children:[z.toFixed(1),"x"]})]})]})]}),f&&r.jsxs("div",{className:"progress-section",children:[r.jsx("div",{className:"progress-bar",children:r.jsx("div",{className:"progress-fill",style:{width:`${P}%`}})}),r.jsxs("div",{className:"progress-status",children:[r.jsx(Oe,{size:16,className:"spin"}),j]})]}),m&&r.jsxs("div",{className:"error-message",children:["⚠️ ",m]}),r.jsx("button",{className:"btn-primary btn-large",onClick:A,disabled:!s.trim()||f,children:f?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{size:18,className:"spin"}),"Generating..."]}):r.jsxs(r.Fragment,{children:[r.jsx(oo,{size:18}),"Generate ",t==="tts"?"Speech":t==="music"?"Music":"Sound"]})}),G&&r.jsxs("div",{className:"result-section",children:[r.jsx("h3",{children:"Result"}),r.jsxs("div",{className:"audio-player",children:[r.jsx("audio",{ref:L,src:G.url,onEnded:()=>C(!1),onPlay:()=>C(!0),onPause:()=>C(!1)}),r.jsx("button",{className:"play-btn",onClick:$,children:N?r.jsx(Fh,{size:24}):r.jsx(si,{size:24})}),r.jsx("div",{className:"audio-info",children:r.jsx("span",{className:"audio-filename",children:G.filename})}),r.jsx("a",{href:G.url,download:!0,className:"download-btn",children:r.jsx(fr,{size:18})})]})]}),r.jsx("style",{children:`
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
      `})]})}const Nc=[{id:"1:1",label:"1:1 (Square)",width:1024,height:1024},{id:"16:9",label:"16:9 (Widescreen)",width:1280,height:720},{id:"9:16",label:"9:16 (Portrait)",width:720,height:1280},{id:"4:3",label:"4:3 (Standard)",width:1024,height:768},{id:"3:4",label:"3:4 (Portrait)",width:768,height:1024},{id:"21:9",label:"21:9 (Ultrawide)",width:1344,height:576},{id:"3:2",label:"3:2 (Photo)",width:1152,height:768},{id:"2:3",label:"2:3 (Photo Portrait)",width:768,height:1152}],Kx=[{id:"center",label:"Center",icon:"⊕"},{id:"top",label:"Top",icon:"⬆️"},{id:"bottom",label:"Bottom",icon:"⬇️"},{id:"left",label:"Left",icon:"⬅️"},{id:"right",label:"Right",icon:"➡️"},{id:"top-left",label:"Top Left",icon:"↖️"},{id:"top-right",label:"Top Right",icon:"↗️"},{id:"bottom-left",label:"Bottom Left",icon:"↙️"},{id:"bottom-right",label:"Bottom Right",icon:"↘️"}],Cc=[{id:"sdxl",label:"SDXL (Quality)",file:"CyberRealisticPony_v8.safetensors"},{id:"flux",label:"Flux (Fast)",file:"flux1-dev-bnb-nf4.safetensors"}];function qx(){const[e,t]=c.useState(null),[n,s]=c.useState(null),[a,l]=c.useState({width:0,height:0}),[o,i]=c.useState(Nc[0]),[d,u]=c.useState("center"),[y,g]=c.useState(Cc[0]),[x,k]=c.useState(""),[S,z]=c.useState(25),[R,f]=c.useState(7),[p,m]=c.useState(.85),[h,j]=c.useState(32),[_,P]=c.useState(!1),[I,G]=c.useState(0),[H,N]=c.useState(null),[C,L]=c.useState(null),[X,A]=c.useState(!1),$=c.useRef(null),O=c.useCallback(v=>{var b,D,V,Y;v.preventDefault();const K=((D=(b=v.dataTransfer)==null?void 0:b.files)==null?void 0:D[0])||((Y=(V=v.target)==null?void 0:V.files)==null?void 0:Y[0]);if(K&&K.type.startsWith("image/")){t(K),N(null),L(null);const F=URL.createObjectURL(K),le=new Image;le.onload=()=>{l({width:le.naturalWidth,height:le.naturalHeight}),s(F)},le.src=F}},[]),M=v=>v.preventDefault(),B=async v=>{var D,V,Y;let b=0;for(;b<300;){await new Promise(le=>setTimeout(le,1e3)),b++,G(Math.min(95,b*.5));const F=await Zu(`${ee}/comfyui/job/${v}`);if(((D=F.data)==null?void 0:D.status)==="completed")return F.data;if(((V=F.data)==null?void 0:V.status)==="error")throw new Error(((Y=F.data)==null?void 0:Y.error)||"Generation failed")}throw new Error("Generation timed out")},Q=async()=>{var v,K,b,D;if(!e){L("Please upload an image first");return}P(!0),G(0),L(null),N(null);try{const V=new FormData;V.append("image",e),V.append("target_width",o.width),V.append("target_height",o.height),V.append("position",d),V.append("prompt",x||"seamless natural extension, high quality"),V.append("model",y.file),V.append("steps",S),V.append("cfg",R),V.append("denoise",p),V.append("feathering",h);const Y=await Nt(`${ee}/reframe`,V);if(!Y.ok)throw new Error(((v=Y.data)==null?void 0:v.detail)||"Reframe request failed");if((K=Y.data)!=null&&K.prompt_id){G(5);const F=await B(Y.data.prompt_id);((b=F.images)==null?void 0:b.length)>0?N({url:F.images[0],prompt_id:Y.data.prompt_id}):F.url&&N({url:F.url,prompt_id:Y.data.prompt_id})}else(D=Y.data)!=null&&D.url&&N({url:Y.data.url});G(100)}catch(V){console.error("❌ Reframe error:",V),L(V.message)}finally{P(!1)}},te=()=>{if(!(H!=null&&H.url))return;const v=document.createElement("a");v.href=H.url,v.download=`reframed_${o.id.replace(":","x")}_${Date.now()}.png`,v.click()},T=(()=>{if(!a.width||!a.height)return null;const v=o.width,K=o.height,b=a.width,D=a.height,V=v/b,Y=K/D,F=Math.min(V,Y),le=Math.round(b*F),re=Math.round(D*F);let ie=0,me=0;return d.includes("left")?ie=0:d.includes("right")?ie=v-le:ie=(v-le)/2,d.includes("top")?me=0:d.includes("bottom")?me=K-re:me=(K-re)/2,{scaledW:le,scaledH:re,offsetX:ie,offsetY:me,targetW:v,targetH:K}})();return r.jsxs("div",{className:"space-y-4",children:[r.jsxs("div",{onClick:()=>{var v;return(v=$.current)==null?void 0:v.click()},onDrop:O,onDragOver:M,className:"border-2 border-dashed border-gray-600 rounded-lg p-6 text-center cursor-pointer hover:border-purple-500 transition-colors",children:[r.jsx("input",{ref:$,type:"file",accept:"image/*",onChange:O,className:"hidden"}),n?r.jsxs("div",{className:"flex flex-col items-center gap-2",children:[r.jsx("img",{src:n,alt:"Preview",className:"max-h-32 rounded"}),r.jsxs("span",{className:"text-sm text-gray-400",children:["Original: ",a.width,"×",a.height]}),r.jsx("span",{className:"text-xs text-gray-500",children:"Click to change"})]}):r.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[r.jsx(St,{className:"w-8 h-8"}),r.jsx("span",{children:"Drop image here or click to upload"})]})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Target Aspect Ratio"}),r.jsx("div",{className:"grid grid-cols-4 gap-2",children:Nc.map(v=>r.jsx("button",{onClick:()=>i(v),className:`px-3 py-2 text-sm rounded transition-colors ${o.id===v.id?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:v.label},v.id))}),r.jsxs("span",{className:"text-xs text-gray-500 mt-1 block",children:["Output: ",o.width,"×",o.height]})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Image Position"}),r.jsx("div",{className:"grid grid-cols-3 gap-2 w-40 mx-auto",children:["top-left","top","top-right","left","center","right","bottom-left","bottom","bottom-right"].map(v=>{var K;return r.jsx("button",{onClick:()=>u(v),className:`p-2 text-lg rounded transition-colors ${d===v?"bg-purple-600":"bg-gray-700 hover:bg-gray-600"}`,title:v,children:((K=Kx.find(b=>b.id===v))==null?void 0:K.icon)||"○"},v)})})]}),T&&r.jsxs("div",{className:"bg-gray-800 rounded-lg p-4",children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Layout Preview"}),r.jsxs("div",{className:"relative mx-auto border border-gray-600 bg-gray-900",style:{width:Math.min(300,T.targetW/3),height:Math.min(300,T.targetH/3),aspectRatio:`${T.targetW} / ${T.targetH}`},children:[r.jsx("div",{className:"absolute inset-0 bg-stripes opacity-30"}),r.jsx("div",{className:"absolute bg-purple-600/50 border-2 border-purple-400 flex items-center justify-center text-xs",style:{width:`${T.scaledW/T.targetW*100}%`,height:`${T.scaledH/T.targetH*100}%`,left:`${T.offsetX/T.targetW*100}%`,top:`${T.offsetY/T.targetH*100}%`},children:"Original"})]}),r.jsx("p",{className:"text-xs text-gray-500 text-center mt-2",children:"Purple = original image, striped = AI-generated fill"})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Fill Prompt (optional)"}),r.jsx("textarea",{value:x,onChange:v=>k(v.target.value),placeholder:"Describe what should appear in the extended areas...",className:"w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 resize-none",rows:2})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Model"}),r.jsx("div",{className:"flex gap-2",children:Cc.map(v=>r.jsx("button",{onClick:()=>g(v),className:`flex-1 px-3 py-2 text-sm rounded transition-colors ${y.id===v.id?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:v.label},v.id))})]}),r.jsxs("div",{className:"border border-gray-700 rounded-lg overflow-hidden",children:[r.jsxs("button",{onClick:()=>A(!X),className:"w-full px-4 py-2 bg-gray-800 flex items-center justify-between text-gray-300 hover:bg-gray-750",children:[r.jsx("span",{className:"text-sm font-medium",children:"Advanced Settings"}),r.jsx(Mt,{className:`w-4 h-4 transition-transform ${X?"rotate-180":""}`})]}),X&&r.jsxs("div",{className:"p-4 space-y-4 bg-gray-850",children:[r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Steps: ",S]}),r.jsx("input",{type:"range",min:10,max:50,value:S,onChange:v=>z(Number(v.target.value)),className:"w-full accent-purple-500"})]}),r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["CFG Scale: ",R]}),r.jsx("input",{type:"range",min:1,max:15,step:.5,value:R,onChange:v=>f(Number(v.target.value)),className:"w-full accent-purple-500"})]}),r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Denoise: ",p.toFixed(2)]}),r.jsx("input",{type:"range",min:.5,max:1,step:.05,value:p,onChange:v=>m(Number(v.target.value)),className:"w-full accent-purple-500"}),r.jsx("span",{className:"text-xs text-gray-500",children:"Higher = more creative fill"})]}),r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Edge Feathering: ",h,"px"]}),r.jsx("input",{type:"range",min:0,max:64,step:8,value:h,onChange:v=>j(Number(v.target.value)),className:"w-full accent-purple-500"}),r.jsx("span",{className:"text-xs text-gray-500",children:"Blend between original and fill"})]})]})]}),r.jsx("button",{onClick:Q,disabled:_||!e,className:"w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors",children:_?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{className:"w-5 h-5 animate-spin"}),"Reframing... ",I>0&&`${Math.round(I)}%`]}):r.jsxs(r.Fragment,{children:[r.jsx(fh,{className:"w-5 h-5"}),"Reframe Image"]})}),C&&r.jsx("div",{className:"p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm",children:C}),H&&r.jsxs("div",{className:"space-y-3",children:[r.jsx("div",{className:"rounded-lg overflow-hidden border border-gray-700",children:r.jsx("img",{src:H.url,alt:"Reframed",className:"w-full"})}),r.jsxs("div",{className:"flex gap-2",children:[r.jsxs("button",{onClick:te,className:"flex-1 py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center justify-center gap-2",children:[r.jsx(fr,{className:"w-4 h-4"}),"Download"]}),r.jsxs("button",{onClick:()=>{t(null),s(null),N(null),fetch(H.url).then(v=>v.blob()).then(v=>{const K=new File([v],"reframed.png",{type:"image/png"});t(K),s(H.url);const b=new Image;b.onload=()=>l({width:b.naturalWidth,height:b.naturalHeight}),b.src=H.url})},className:"flex-1 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center justify-center gap-2",children:[r.jsx(Ih,{className:"w-4 h-4"}),"Use as Input"]})]})]}),r.jsxs("div",{className:"text-xs text-gray-500 space-y-1",children:[r.jsxs("p",{children:["💡 ",r.jsx("strong",{children:"Reframe"})," extends your image to a new aspect ratio using AI outpainting."]}),r.jsx("p",{children:"📐 The original image will be placed according to the position you select."}),r.jsx("p",{children:"🎨 Use the prompt to guide what should appear in the extended areas."})]})]})}const _c=[{id:"inswapper",label:"InSwapper (Best Quality)",description:"High quality, slower"},{id:"simswap",label:"SimSwap (Fast)",description:"Faster, good quality"}],Jx=[{id:"none",label:"None"},{id:"gfpgan",label:"GFPGAN (Faces)"},{id:"codeformer",label:"CodeFormer (Natural)"},{id:"both",label:"Both (Best)"}];function Zx(){const[e,t]=c.useState(null),[n,s]=c.useState(null),[a,l]=c.useState(null),[o,i]=c.useState(null),[d,u]=c.useState(_c[0]),[y,g]=c.useState("gfpgan"),[x,k]=c.useState(1),[S,z]=c.useState(.8),[R,f]=c.useState(0),[p,m]=c.useState(!1),[h,j]=c.useState(!1),[_,P]=c.useState(0),[I,G]=c.useState(null),[H,N]=c.useState(null),[C,L]=c.useState(null),[X,A]=c.useState(!1),$=c.useRef(null),O=c.useRef(null),M=c.useCallback(b=>{var V,Y,F,le;b.preventDefault();const D=((Y=(V=b.dataTransfer)==null?void 0:V.files)==null?void 0:Y[0])||((le=(F=b.target)==null?void 0:F.files)==null?void 0:le[0]);if(D&&(D.type.startsWith("image/")||D.type.startsWith("video/"))){t(D),G(null),N(null),L(null);const re=URL.createObjectURL(D);s(re)}},[]),B=c.useCallback(b=>{var V,Y,F,le;b.preventDefault();const D=((Y=(V=b.dataTransfer)==null?void 0:V.files)==null?void 0:Y[0])||((le=(F=b.target)==null?void 0:F.files)==null?void 0:le[0]);if(D&&D.type.startsWith("image/")){l(D),G(null),N(null);const re=URL.createObjectURL(D);i(re)}},[]),Q=b=>b.preventDefault(),te=async()=>{var b,D;if(e){j(!0),N(null);try{const V=new FormData;V.append("image",e);const Y=await Nt(`${ee}/detect-faces`,V);if(Y.ok&&((b=Y.data)!=null&&b.faces))L(Y.data.faces);else throw new Error(((D=Y.data)==null?void 0:D.detail)||"Face detection failed")}catch(V){console.error("❌ Face detection error:",V),N(V.message)}finally{j(!1)}}},oe=async b=>{var Y,F,le;let V=0;for(;V<300;){await new Promise(ie=>setTimeout(ie,1e3)),V++,P(Math.min(95,V*.5));const re=await Zu(`${ee}/comfyui/job/${b}`);if(((Y=re.data)==null?void 0:Y.status)==="completed")return re.data;if(((F=re.data)==null?void 0:F.status)==="error")throw new Error(((le=re.data)==null?void 0:le.error)||"Face swap failed")}throw new Error("Face swap timed out")},T=async()=>{var b,D,V,Y;if(!e||!a){N("Please upload both target and source face images");return}j(!0),P(0),N(null),G(null);try{const F=new FormData;F.append("target",e),F.append("source",a),F.append("model",d.id),F.append("enhance",y),F.append("strength",x),F.append("blend",S),F.append("face_index",p?-1:R);const le=await Nt(`${ee}/face-swap`,F);if(!le.ok)throw new Error(((b=le.data)==null?void 0:b.detail)||"Face swap request failed");if((D=le.data)!=null&&D.prompt_id){P(5);const re=await oe(le.data.prompt_id);((V=re.images)==null?void 0:V.length)>0?G({url:re.images[0],prompt_id:le.data.prompt_id}):re.url&&G({url:re.url,prompt_id:le.data.prompt_id})}else(Y=le.data)!=null&&Y.url&&G({url:le.data.url});P(100)}catch(F){console.error("❌ FaceSwap error:",F),N(F.message)}finally{j(!1)}},v=()=>{if(!(I!=null&&I.url))return;const b=e!=null&&e.type.startsWith("video/")?"mp4":"png",D=document.createElement("a");D.href=I.url,D.download=`face_swap_${Date.now()}.${b}`,D.click()},K=()=>{const b=e,D=n;t(a),s(o),l(b),i(D),G(null),L(null)};return r.jsxs("div",{className:"space-y-4",children:[r.jsxs("div",{className:"grid grid-cols-2 gap-4",children:[r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Target (face to replace)"}),r.jsxs("div",{onClick:()=>{var b;return(b=$.current)==null?void 0:b.click()},onDrop:M,onDragOver:Q,className:"border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-purple-500 transition-colors aspect-square flex items-center justify-center",children:[r.jsx("input",{ref:$,type:"file",accept:"image/*,video/*",onChange:M,className:"hidden"}),n?r.jsxs("div",{className:"relative w-full h-full",children:[e!=null&&e.type.startsWith("video/")?r.jsx("video",{src:n,className:"w-full h-full object-cover rounded",muted:!0}):r.jsx("img",{src:n,alt:"Target",className:"w-full h-full object-cover rounded"}),C&&r.jsxs("div",{className:"absolute bottom-1 right-1 bg-black/70 px-2 py-1 rounded text-xs",children:[C.length," face",C.length!==1?"s":""," detected"]})]}):r.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[r.jsx(St,{className:"w-6 h-6"}),r.jsx("span",{className:"text-xs",children:"Target image/video"})]})]})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Source (face to use)"}),r.jsxs("div",{onClick:()=>{var b;return(b=O.current)==null?void 0:b.click()},onDrop:B,onDragOver:Q,className:"border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-blue-500 transition-colors aspect-square flex items-center justify-center",children:[r.jsx("input",{ref:O,type:"file",accept:"image/*",onChange:B,className:"hidden"}),o?r.jsx("img",{src:o,alt:"Source",className:"w-full h-full object-cover rounded"}):r.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[r.jsx(Gh,{className:"w-6 h-6"}),r.jsx("span",{className:"text-xs",children:"Source face"})]})]})]})]}),(e||a)&&r.jsxs("button",{onClick:K,className:"w-full py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center justify-center gap-2 text-sm",children:[r.jsx(dn,{className:"w-4 h-4"}),"Swap Target ↔ Source"]}),e&&!e.type.startsWith("video/")&&r.jsxs("button",{onClick:te,disabled:h,className:"w-full py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center justify-center gap-2 text-sm",children:[r.jsx(lo,{className:"w-4 h-4"}),"Detect Faces"]}),C&&C.length>1&&r.jsxs("div",{className:"bg-gray-800 rounded-lg p-3 space-y-2",children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300",children:"Select Face to Replace"}),r.jsx("div",{className:"flex items-center gap-4",children:r.jsxs("label",{className:"flex items-center gap-2",children:[r.jsx("input",{type:"checkbox",checked:p,onChange:b=>m(b.target.checked),className:"rounded bg-gray-700 border-gray-600"}),r.jsx("span",{className:"text-sm text-gray-300",children:"Swap all faces"})]})}),!p&&r.jsx("div",{className:"flex gap-2 flex-wrap",children:C.map((b,D)=>r.jsxs("button",{onClick:()=>f(D),className:`px-3 py-1 text-sm rounded ${R===D?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:["Face ",D+1]},D))})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Model"}),r.jsx("div",{className:"space-y-2",children:_c.map(b=>r.jsxs("button",{onClick:()=>u(b),className:`w-full px-3 py-2 text-left rounded transition-colors ${d.id===b.id?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:[r.jsx("div",{className:"font-medium text-sm",children:b.label}),r.jsx("div",{className:"text-xs opacity-70",children:b.description})]},b.id))})]}),r.jsxs("div",{children:[r.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Face Enhancement"}),r.jsx("div",{className:"grid grid-cols-2 gap-2",children:Jx.map(b=>r.jsx("button",{onClick:()=>g(b.id),className:`px-3 py-2 text-sm rounded transition-colors ${y===b.id?"bg-blue-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:b.label},b.id))})]}),r.jsxs("div",{className:"border border-gray-700 rounded-lg overflow-hidden",children:[r.jsxs("button",{onClick:()=>A(!X),className:"w-full px-4 py-2 bg-gray-800 flex items-center justify-between text-gray-300 hover:bg-gray-750",children:[r.jsx("span",{className:"text-sm font-medium",children:"Advanced Settings"}),r.jsx(Mt,{className:`w-4 h-4 transition-transform ${X?"rotate-180":""}`})]}),X&&r.jsxs("div",{className:"p-4 space-y-4 bg-gray-850",children:[r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Swap Strength: ",x.toFixed(2)]}),r.jsx("input",{type:"range",min:.1,max:1,step:.05,value:x,onChange:b=>k(Number(b.target.value)),className:"w-full accent-purple-500"}),r.jsx("span",{className:"text-xs text-gray-500",children:"Lower = more original features preserved"})]}),r.jsxs("div",{children:[r.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Edge Blend: ",S.toFixed(2)]}),r.jsx("input",{type:"range",min:0,max:1,step:.05,value:S,onChange:b=>z(Number(b.target.value)),className:"w-full accent-purple-500"}),r.jsx("span",{className:"text-xs text-gray-500",children:"Blend face edges with background"})]})]})]}),r.jsxs("div",{className:"flex items-start gap-2 p-3 bg-yellow-900/30 border border-yellow-700/50 rounded-lg",children:[r.jsx(Hm,{className:"w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5"}),r.jsxs("div",{className:"text-sm text-yellow-200",children:[r.jsx("strong",{children:"Ethical Use:"})," Only use face swap with consent of all parties involved. Creating non-consensual deepfakes is illegal in many jurisdictions."]})]}),r.jsx("button",{onClick:T,disabled:h||!e||!a,className:"w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors",children:h?r.jsxs(r.Fragment,{children:[r.jsx(Oe,{className:"w-5 h-5 animate-spin"}),"Swapping... ",_>0&&`${Math.round(_)}%`]}):r.jsxs(r.Fragment,{children:[r.jsx(lo,{className:"w-5 h-5"}),"Swap Face"]})}),H&&r.jsx("div",{className:"p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm",children:H}),I&&r.jsxs("div",{className:"space-y-3",children:[r.jsx("div",{className:"rounded-lg overflow-hidden border border-gray-700",children:e!=null&&e.type.startsWith("video/")?r.jsx("video",{src:I.url,controls:!0,className:"w-full"}):r.jsx("img",{src:I.url,alt:"Result",className:"w-full"})}),r.jsxs("button",{onClick:v,className:"w-full py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center justify-center gap-2",children:[r.jsx(fr,{className:"w-4 h-4"}),"Download Result"]})]}),r.jsxs("div",{className:"text-xs text-gray-500 space-y-1",children:[r.jsxs("p",{children:["👤 ",r.jsx("strong",{children:"Face Swap"})," replaces faces in images or videos using AI."]}),r.jsx("p",{children:"📸 For best results, use clear frontal face photos with good lighting."}),r.jsx("p",{children:"🎬 Video processing may take longer depending on length and resolution."})]})]})}function eg({title:e}){return r.jsxs("div",{className:"tool-coming-soon",children:[r.jsx("div",{className:"tool-title",children:e}),r.jsx("div",{className:"muted",children:"Missing backend endpoint (planned for v2)."})]})}const Ec=e=>{if(!e||isNaN(e))return null;const t=Math.floor(e/60),n=Math.floor(e%60);return`${t}:${n.toString().padStart(2,"0")}`},ep="oelala_media_favorites",tp="oelala_media_profile",Rs={"1280x1024":{cols:4,label:"1280×1024"},"1080p":{cols:5,label:"1080p"},"1440p":{cols:6,label:"1440p"},"4k":{cols:8,label:"4K"}},zc=()=>{const e=window.innerWidth;return e<=1280?"1280x1024":e<=1920?"1080p":e<=2560?"1440p":"4k"},tg=()=>{try{return localStorage.getItem(tp)||"auto"}catch{return"auto"}},pl=e=>{try{localStorage.setItem(tp,e)}catch(t){console.error("Failed to save profile:",t)}},rg=()=>{try{const e=localStorage.getItem(ep);return e?new Set(JSON.parse(e)):new Set}catch{return new Set}},ng=e=>{try{localStorage.setItem(ep,JSON.stringify([...e]))}catch(t){console.error("Failed to save favorites:",t)}};function zn({filter:e="all",selectionMode:t=!1,onSelectItem:n=null}){var fs,hn,ms,Fa,hs,jr,xn,gn,xt,Da,Fr,xs,vn;const[s,a]=c.useState([]),[l,o]=c.useState(!1),[i,d]=c.useState(""),[u,y]=c.useState({videos:0,images:0}),[g,x]=c.useState(null),[k,S]=c.useState(new Set),[z,R]=c.useState(null),[f,p]=c.useState(!1),[m,h]=c.useState(!1),[j,_]=c.useState(null),[P,I]=c.useState(rg),[G,H]=c.useState("date"),[N,C]=c.useState("desc"),[L,X]=c.useState("all"),[A,$]=c.useState(!0),[O,M]=c.useState(tg),B=O==="auto"?zc():O,te=(Rs[B]||Rs["1080p"]).cols,[oe,T]=c.useState(!1),[v,K]=c.useState(100),[b,D]=c.useState(320),[V,Y]=c.useState({}),F=c.useRef(null);c.useEffect(()=>{const w=()=>{if(F.current){const ae=(F.current.clientWidth-32-12*(te-1))/te,ve=Math.round(ae*(16/9));D(ve)}};return w(),window.addEventListener("resize",w),()=>window.removeEventListener("resize",w)},[te]),c.useEffect(()=>{K(100)},[L,G,N,s]);const le=w=>{const{scrollTop:W,clientHeight:J,scrollHeight:ae}=w.target;ae-W-J<1e3&&K(ve=>Math.min(ve+50,ie.length))},re=(w,W)=>{W==null||W.stopPropagation(),I(J=>{const ae=new Set(J);return ae.has(w)?ae.delete(w):ae.add(w),ng(ae),ae})},ie=c.useMemo(()=>{let w=[...s];return L==="favorites"?w=w.filter(W=>P.has(W.filename)):L==="non-favorites"&&(w=w.filter(W=>!P.has(W.filename))),w.sort((W,J)=>{let ae=0;switch(G){case"name":ae=W.filename.localeCompare(J.filename);break;case"size":ae=(W.size||0)-(J.size||0);break;case"favorites":const ve=P.has(W.filename)?1:0,ce=P.has(J.filename)?1:0;ae=ve-ce;break;case"non-favorites":const he=P.has(W.filename)?0:1,Yt=P.has(J.filename)?0:1;ae=he-Yt;break;case"date":default:ae=(W.mtime||0)-(J.mtime||0);break}return N==="desc"?-ae:ae}),w},[s,G,N,L,P]),me=c.useCallback(async()=>{o(!0),d("");try{const W=await fetch(`${ee}/list-comfyui-media?type=${e==="prompts"?"all":e}&grouped=true&include_metadata=true&hide_start_images=${A}`);if(!W.ok)throw new Error("Failed to fetch media");const J=await W.json();let ae=J.media||[];e==="prompts"&&(ae=ae.filter(ve=>{var ce,he;return((ce=ve.metadata)==null?void 0:ce.positive_prompt)||((he=ve.metadata)==null?void 0:he.prompt)})),a(ae),y({videos:J.videos||0,images:J.images||0}),S(new Set)}catch(w){d(w.message)}finally{o(!1)}},[e,A]);c.useEffect(()=>{me()},[me]),c.useEffect(()=>{const w=W=>{if(W.key==="?"||W.key==="/"&&W.shiftKey){W.preventDefault(),T(J=>!J);return}if(W.key==="+"||W.key==="="){W.preventDefault();const J=["auto","1280x1024","1080p","1440p","4k"];M(ae=>{const ve=J.indexOf(ae),ce=J[(ve+1)%J.length];return pl(ce),ce});return}if(W.key==="-"||W.key==="_"){W.preventDefault();const J=["auto","1280x1024","1080p","1440p","4k"];M(ae=>{const ve=J.indexOf(ae),ce=J[(ve-1+J.length)%J.length];return pl(ce),ce});return}if(g!==null&&(W.key==="Escape"&&(x(null),T(!1)),W.key==="ArrowLeft"&&x(J=>J>0?J-1:ie.length-1),W.key==="ArrowRight"&&x(J=>J<ie.length-1?J+1:0),W.key==="f"||W.key==="F"||W.key==="h"||W.key==="H")){const J=ie[g];J&&re(J.filename)}};return window.addEventListener("keydown",w),()=>{window.removeEventListener("keydown",w)}},[g,ie,P]);const Ze=(w,W)=>{if(W.target.closest(".select-checkbox")){W.stopPropagation(),Re(w,W);return}if(t&&n){const J=ie[w];n(J);return}x(w)},Re=(w,W)=>{W==null||W.stopPropagation(),S(J=>{const ae=new Set(J);if(W!=null&&W.shiftKey&&z!==null){const ve=Math.min(z,w),ce=Math.max(z,w);for(let he=ve;he<=ce;he++)ae.add(he)}else W!=null&&W.ctrlKey||W!=null&&W.metaKey,ae.has(w)?ae.delete(w):ae.add(w);return ae}),R(w)},ht=()=>{S(new Set(s.map((w,W)=>W)))},Xt=()=>{S(new Set)},Lr=async()=>{if(k.size===0)return;const w=Array.from(k).map(ce=>{var he;return(he=ie[ce])==null?void 0:he.filename}).filter(Boolean);if(w.length===0){d("No valid items selected for deletion");return}const W=w.filter(ce=>P.has(ce)),J=W.length;let ae=`Delete ${w.length} item${w.length>1?"s":""} and their associated files (source images, metadata)?`;if(J>0&&(ae=`⚠️ WARNING: ${J} favorite${J>1?"s":""} selected!

${ae}

Favorites to delete:
• ${W.slice(0,5).join(`
• `)}${J>5?`
• ... and ${J-5} more`:""}`),!!window.confirm(ae)){p(!0);try{const ce=await fetch(`${ee}/delete-comfyui-media`,{method:"DELETE",headers:{"Content-Type":"application/json"},body:JSON.stringify({filenames:w})});if(!ce.ok)throw new Error("Failed to delete");const he=await ce.json();console.log("Deleted:",he),await me()}catch(ce){d(`Delete failed: ${ce.message}`)}finally{p(!1)}}},et=(w,W)=>{W==null||W.stopPropagation();const J=document.createElement("a");J.href=`${ee}${w.url}`,J.download=w.filename,J.click()},yr=async(w,W)=>{W==null||W.stopPropagation();try{const J=await fetch(`${ee}/comfyui-metadata/${w.filename}`);if(!J.ok)throw new Error("No metadata available");const ae=await J.json(),ve=new Blob([JSON.stringify(ae.metadata,null,2)],{type:"application/json"}),ce=URL.createObjectURL(ve),he=document.createElement("a");he.href=ce,he.download=`${w.base_name||w.filename.replace(/\.[^/.]+$/,"")}_metadata.json`,he.click(),URL.revokeObjectURL(ce)}catch(J){console.error("Failed to download metadata:",J)}},Ct=w=>w<1024?`${w} B`:w<1024*1024?`${(w/1024).toFixed(1)} KB`:`${(w/1024/1024).toFixed(1)} MB`,xe=g!==null?ie[g]:null,mn=s.filter(w=>P.has(w.filename)).length;return r.jsxs("div",{style:{display:"flex",flexDirection:"column",height:"100%",backgroundColor:"var(--bg-primary)"},children:[r.jsx("style",{children:`
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
      `}),r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"12px 16px",borderBottom:"1px solid var(--border-color)",backgroundColor:"var(--bg-secondary)",flexWrap:"wrap",gap:"10px"},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"16px"},children:[r.jsx("span",{style:{fontWeight:600,color:"var(--text-primary)"},children:e==="all"?"All Media":e==="video"?"Videos":e==="image"?"Images":"Prompts"}),r.jsxs("span",{style:{color:"var(--text-muted)",fontSize:"0.85rem"},children:[e==="prompts"?r.jsxs(r.Fragment,{children:["💬 ",ie.length," items with prompts"]}):r.jsxs(r.Fragment,{children:["🎬 ",u.videos," • 🖼️ ",u.images," • ❤️ ",mn]}),L!=="all"&&` • 📋 ${ie.length} shown`]})]}),r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"4px"},children:[r.jsx(dh,{size:14,style:{color:"var(--text-muted)"}}),r.jsxs("select",{className:"sort-select",value:L,onChange:w=>{X(w.target.value),S(new Set)},children:[r.jsx("option",{value:"all",children:"All"}),r.jsx("option",{value:"favorites",children:"❤️ Favorites"}),r.jsx("option",{value:"non-favorites",children:"🤍 Non-favorites"})]}),(e==="all"||e==="image")&&r.jsxs("button",{className:"sort-btn",onClick:()=>$(w=>!w),title:A?"Click to show video source images":"Hiding video source images",style:{background:A?void 0:"var(--accent-color, #a855f7)",color:A?void 0:"#fff",fontSize:"0.75rem",padding:"4px 8px"},children:["📸",A?"":"✓"]})]}),r.jsx("div",{style:{width:"1px",height:"20px",background:"var(--border-color)",margin:"0 4px"}}),r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"4px"},children:[r.jsx(Om,{size:14,style:{color:"var(--text-muted)"}}),r.jsxs("select",{className:"sort-select",value:G,onChange:w=>H(w.target.value),children:[r.jsx("option",{value:"date",children:"Date"}),r.jsx("option",{value:"name",children:"Name"}),r.jsx("option",{value:"size",children:"Size"}),r.jsx("option",{value:"favorites",children:"Favorites ❤️"}),r.jsx("option",{value:"non-favorites",children:"Non-favorites 🤍"})]}),r.jsx("button",{className:"sort-btn",onClick:()=>C(w=>w==="asc"?"desc":"asc"),title:N==="asc"?"Ascending":"Descending",children:N==="asc"?"↑":"↓"})]}),r.jsx("div",{style:{width:"1px",height:"20px",background:"var(--border-color)",margin:"0 4px"}}),r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"2px"},children:[r.jsx("span",{style:{color:"var(--text-muted)",fontSize:"0.75rem",marginRight:"4px"},children:"Profile:"}),["auto","1280x1024","1080p","1440p","4k"].map(w=>{var W,J;return r.jsx("button",{className:"sort-btn",onClick:()=>{M(w),pl(w)},title:w==="auto"?`Auto-detect (currently ${zc()})`:((W=Rs[w])==null?void 0:W.label)||w,style:{background:O===w?"var(--accent-color, #a855f7)":void 0,color:O===w?"#fff":void 0,fontSize:"0.7rem",padding:"4px 6px"},children:w==="auto"?"⚡Auto":((J=Rs[w])==null?void 0:J.label)||w},w)}),r.jsxs("span",{style:{color:"var(--text-muted)",fontSize:"0.7rem",marginLeft:"8px"},children:[te," cols"]})]}),r.jsx("div",{style:{width:"1px",height:"20px",background:"var(--border-color)",margin:"0 4px"}}),k.size>0&&r.jsxs(r.Fragment,{children:[r.jsxs("span",{style:{color:"var(--text-muted)",fontSize:"0.85rem"},children:[k.size," selected"]}),r.jsx("button",{className:"header-btn",onClick:Xt,children:"Clear"}),r.jsx("button",{className:"header-btn",onClick:ht,children:"Select All"}),r.jsxs("button",{className:"delete-btn",onClick:Lr,disabled:f,children:[r.jsx(Yh,{size:16}),f?"Deleting...":"Delete"]})]}),r.jsx("button",{onClick:me,disabled:l,style:{padding:"8px",borderRadius:"6px",border:"none",background:"transparent",color:"var(--text-muted)",cursor:"pointer",display:"flex",alignItems:"center"},title:"Refresh",children:r.jsx(dn,{size:18,style:{animation:l?"spin 1s linear infinite":"none"}})}),r.jsx("button",{onClick:()=>T(!0),style:{padding:"6px",border:"none",background:"transparent",color:"var(--text-muted)",cursor:"pointer",display:"flex",alignItems:"center"},title:"Keyboard shortcuts (?)",children:r.jsx(Hu,{size:18})})]})]}),oe&&r.jsx("div",{style:{position:"fixed",top:0,left:0,right:0,bottom:0,backgroundColor:"rgba(0,0,0,0.8)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:2e3},onClick:()=>T(!1),children:r.jsxs("div",{style:{backgroundColor:"var(--bg-primary, #1a1a1a)",borderRadius:"12px",padding:"24px",maxWidth:"500px",width:"90%",boxShadow:"0 20px 60px rgba(0,0,0,0.5)"},onClick:w=>w.stopPropagation(),children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"20px"},children:[r.jsx("h3",{style:{margin:0,color:"var(--text-primary, #fff)",fontSize:"1.2rem"},children:"⌨️ Keyboard Shortcuts"}),r.jsx("button",{onClick:()=>T(!1),style:{background:"transparent",border:"none",color:"var(--text-muted)",cursor:"pointer",padding:"4px"},children:r.jsx(It,{size:20})})]}),r.jsxs("div",{style:{color:"var(--text-secondary, #ccc)",fontSize:"0.9rem"},children:[r.jsxs("div",{style:{marginBottom:"16px"},children:[r.jsx("div",{style:{color:"var(--accent-color, #a855f7)",fontWeight:600,marginBottom:"8px"},children:"Grid View"}),r.jsxs("div",{style:{display:"grid",gridTemplateColumns:"120px 1fr",gap:"6px 16px"},children:[r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"+"}),r.jsx("span",{children:"More columns (smaller thumbnails)"}),r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"-"}),r.jsx("span",{children:"Fewer columns (larger thumbnails)"}),r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"?"}),r.jsx("span",{children:"Show this help"})]})]}),r.jsxs("div",{style:{marginBottom:"16px"},children:[r.jsx("div",{style:{color:"var(--accent-color, #a855f7)",fontWeight:600,marginBottom:"8px"},children:"Lightbox (Image View)"}),r.jsxs("div",{style:{display:"grid",gridTemplateColumns:"120px 1fr",gap:"6px 16px"},children:[r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"←"}),r.jsx("span",{children:"Previous image"}),r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"→"}),r.jsx("span",{children:"Next image"}),r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"F / H"}),r.jsx("span",{children:"Toggle favorite ❤️"}),r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"Esc"}),r.jsx("span",{children:"Close lightbox"})]})]}),r.jsxs("div",{children:[r.jsx("div",{style:{color:"var(--accent-color, #a855f7)",fontWeight:600,marginBottom:"8px"},children:"Selection"}),r.jsxs("div",{style:{display:"grid",gridTemplateColumns:"120px 1fr",gap:"6px 16px"},children:[r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"Ctrl+Click"}),r.jsx("span",{children:"Toggle single item"}),r.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"Shift+Click"}),r.jsx("span",{children:"Select range"})]})]})]}),r.jsx("div",{style:{marginTop:"20px",paddingTop:"16px",borderTop:"1px solid var(--border-color, #333)",textAlign:"center"},children:r.jsxs("span",{style:{color:"var(--text-muted)",fontSize:"0.8rem"},children:["Press ",r.jsx("kbd",{style:{background:"#333",padding:"2px 6px",borderRadius:"4px"},children:"?"})," or ",r.jsx("kbd",{style:{background:"#333",padding:"2px 6px",borderRadius:"4px"},children:"Esc"})," to close"]})})]})}),i&&r.jsx("div",{style:{padding:"12px 16px",backgroundColor:"rgba(239, 68, 68, 0.1)",color:"#ef4444",textAlign:"center"},children:i}),l&&r.jsxs("div",{style:{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",color:"var(--text-muted)"},children:[r.jsx(dn,{size:40,style:{animation:"spin 1s linear infinite",marginBottom:"16px"}}),r.jsx("div",{children:"Loading media..."})]}),!l&&s.length===0&&r.jsxs("div",{style:{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",color:"var(--text-muted)"},children:[r.jsx("div",{style:{fontSize:"4rem",marginBottom:"16px",opacity:.5},children:"📁"}),r.jsxs("div",{style:{fontSize:"1.2rem",marginBottom:"8px"},children:["No ",e==="prompts"?"prompts":e==="all"?"media":e+"s"," yet"]}),r.jsx("div",{style:{fontSize:"0.9rem",opacity:.7},children:"Generated content will appear here"})]}),!l&&ie.length>0&&e==="prompts"&&r.jsx("div",{ref:F,className:"prompts-list",onScroll:le,style:{flex:1,overflowY:"auto",overflowX:"hidden",padding:"16px",display:"flex",flexDirection:"column",gap:"12px"},children:ie.slice(0,v).map((w,W)=>{var J,ae,ve,ce;return r.jsxs("div",{style:{display:"flex",gap:"16px",padding:"16px",backgroundColor:"var(--bg-secondary, #1f1f1f)",borderRadius:"12px",border:"1px solid var(--border-color, #333)",cursor:"pointer",transition:"border-color 0.15s"},onClick:()=>x(W),onMouseEnter:he=>he.currentTarget.style.borderColor="var(--accent-color, #a855f7)",onMouseLeave:he=>he.currentTarget.style.borderColor="var(--border-color, #333)",children:[r.jsx("div",{style:{flexShrink:0},children:w.type==="video"?r.jsx("video",{src:`${ee}${w.url}`,style:{width:"100px",height:"100px",objectFit:"cover",borderRadius:"8px"},autoPlay:!0,loop:!0,muted:!0,playsInline:!0}):r.jsx("img",{src:`${ee}${w.url}`,alt:w.filename,style:{width:"100px",height:"100px",objectFit:"cover",borderRadius:"8px"},loading:"lazy"})}),r.jsxs("div",{style:{flex:1,minWidth:0},children:[r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:"8px"},children:[r.jsxs("div",{children:[r.jsx("div",{style:{fontSize:"0.85rem",fontWeight:600,color:"var(--text-primary)",marginBottom:"4px"},children:w.filename}),r.jsxs("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:[w.type==="video"?"🎬":"🖼️"," ",Ct(w.size),((J=w.metadata)==null?void 0:J.steps)&&` • ${w.metadata.steps} steps`,((ae=w.metadata)==null?void 0:ae.cfg)&&` • CFG ${w.metadata.cfg}`]})]}),r.jsxs("div",{style:{display:"flex",gap:"8px"},children:[r.jsxs("button",{style:{background:"var(--accent-color, #a855f7)",border:"none",color:"#fff",padding:"6px 12px",borderRadius:"6px",cursor:"pointer",fontSize:"0.75rem",display:"flex",alignItems:"center",gap:"4px"},onClick:he=>{var E,Z;he.stopPropagation();const Yt=((E=w.metadata)==null?void 0:E.positive_prompt)||((Z=w.metadata)==null?void 0:Z.prompt);navigator.clipboard.writeText(Yt)},children:[r.jsx(At,{size:12}),"Copy"]}),r.jsx("button",{className:(P.has(w.filename),""),style:{background:P.has(w.filename)?"#ef4444":"rgba(255,255,255,0.1)",border:"none",color:"#fff",padding:"6px",borderRadius:"6px",cursor:"pointer"},onClick:he=>re(w.filename,he),children:r.jsx(dl,{size:14,fill:P.has(w.filename)?"#fff":"none"})})]})]}),r.jsx("div",{style:{fontSize:"0.9rem",color:"var(--text-primary)",lineHeight:1.5,backgroundColor:"var(--bg-tertiary, #2a2a2a)",padding:"10px 12px",borderRadius:"6px",maxHeight:"100px",overflow:"hidden",textOverflow:"ellipsis",display:"-webkit-box",WebkitLineClamp:4,WebkitBoxOrient:"vertical"},children:((ve=w.metadata)==null?void 0:ve.positive_prompt)||((ce=w.metadata)==null?void 0:ce.prompt)})]})]},w.filename)})}),!l&&ie.length>0&&e!=="prompts"&&r.jsx("div",{ref:F,className:"media-grid",onScroll:le,style:{flex:1,overflowY:"auto",overflowX:"hidden",gridTemplateColumns:`repeat(${te}, 1fr)`},children:ie.slice(0,v).map((w,W)=>{var J,ae,ve;return r.jsxs("div",{className:`thumb-card ${k.has(W)?"selected":""}`,style:{height:`${b}px`},onClick:ce=>Ze(W,ce),children:[r.jsx("div",{className:"select-checkbox",onClick:ce=>Re(W,ce),children:k.has(W)&&r.jsx(Bu,{size:14,color:"#fff"})}),r.jsx("div",{className:`favorite-btn ${P.has(w.filename)?"is-favorite":""}`,onClick:ce=>re(w.filename,ce),title:P.has(w.filename)?"Remove from favorites":"Add to favorites",children:r.jsx(dl,{size:14,color:P.has(w.filename)?"#fff":"rgba(255,255,255,0.7)",fill:P.has(w.filename)?"#fff":"none"})}),(((J=w.metadata)==null?void 0:J.positive_prompt)||((ae=w.metadata)==null?void 0:ae.prompt))&&r.jsx("button",{className:"prompt-bubble-btn",onClick:ce=>{ce.stopPropagation(),_({item:w})},title:"View prompt",children:"💬"}),w.has_source_image&&r.jsxs("div",{className:"source-image-badge",children:[r.jsx(mr,{size:10}),r.jsx("span",{children:"+IMG"})]}),w.type==="video"?r.jsx("video",{src:`${ee}${w.url}`,autoPlay:!0,loop:!0,muted:!0,playsInline:!0,preload:"metadata",onLoadedMetadata:ce=>{const he=ce.target.duration;he&&!V[w.filename]&&Y(Yt=>({...Yt,[w.filename]:he}))}}):r.jsx("img",{src:`${ee}${w.url}`,alt:w.filename,loading:"lazy"}),r.jsxs("div",{className:"media-overlay",children:[r.jsxs("div",{children:[r.jsx("div",{className:"media-filename",children:w.filename}),r.jsxs("div",{className:"media-size",children:[Ct(w.size),w.type==="video"&&V[w.filename]&&r.jsxs("span",{className:"media-duration",children:[r.jsx(Ra,{size:10}),Ec(V[w.filename])]})]})]}),r.jsxs("div",{className:"overlay-buttons",children:[((ve=w.metadata)==null?void 0:ve.has_metadata)&&r.jsx("button",{className:"overlay-btn",onClick:ce=>yr(w,ce),title:"Download metadata JSON",children:r.jsx(vc,{size:14})}),r.jsx("button",{className:"overlay-btn",onClick:ce=>et(w,ce),title:"Download",children:r.jsx(fr,{size:14})})]})]})]},w.filename)})}),xe&&r.jsxs("div",{className:"lightbox-overlay",onClick:()=>x(null),children:[r.jsx("button",{className:"lightbox-close",onClick:()=>x(null),children:r.jsx(It,{size:24})}),((fs=xe.metadata)==null?void 0:fs.has_metadata)&&r.jsx("button",{style:{position:"absolute",top:"20px",left:"20px",padding:"8px 12px",borderRadius:"6px",background:m?"var(--accent-color, #a855f7)":"rgba(255,255,255,0.1)",border:"none",color:"#fff",cursor:"pointer",fontSize:"0.85rem",zIndex:1002},onClick:w=>{w.stopPropagation(),h(!m)},children:m?"Hide Prompt":"Show Prompt"}),m&&xe.metadata&&r.jsxs("div",{className:"lightbox-metadata",onClick:w=>w.stopPropagation(),children:[xe.metadata.positive_prompt&&r.jsxs("div",{style:{marginBottom:"16px"},children:[r.jsx("div",{className:"prompt-label",children:"✨ Positive Prompt"}),r.jsx("div",{className:"prompt-text",children:xe.metadata.positive_prompt})]}),xe.metadata.negative_prompt&&r.jsxs("div",{children:[r.jsx("div",{className:"prompt-label",children:"🚫 Negative Prompt"}),r.jsx("div",{className:"prompt-text",style:{color:"rgba(255,255,255,0.6)"},children:xe.metadata.negative_prompt})]})]}),r.jsx("button",{className:"lightbox-nav",style:{left:"20px"},onClick:w=>{w.stopPropagation(),x(W=>W>0?W-1:ie.length-1)},children:r.jsx(Wu,{size:28})}),r.jsx("div",{className:"lightbox-content",onClick:w=>w.stopPropagation(),children:xe.type==="video"?r.jsx("video",{src:`${ee}${xe.url}`,autoPlay:!0,loop:!0,controls:!0,style:{borderRadius:"12px"}}):r.jsx("img",{src:`${ee}${xe.url}`,alt:xe.filename,style:{borderRadius:"12px"}})}),r.jsx("button",{className:"lightbox-nav",style:{right:"20px"},onClick:w=>{w.stopPropagation(),x(W=>W<ie.length-1?W+1:0)},children:r.jsx(Gu,{size:28})}),r.jsxs("div",{className:"lightbox-info",children:[r.jsx("span",{style:{color:"#fff",fontWeight:500},children:xe.filename}),r.jsx("span",{style:{color:"rgba(255,255,255,0.6)"},children:Ct(xe.size)}),P.has(xe.filename)&&r.jsx("span",{style:{color:"#ef4444",fontSize:"0.8rem"},children:"❤️ Favorite"}),xe.has_source_image&&r.jsx("span",{style:{color:"#3b82f6",fontSize:"0.8rem"},children:"📷 Has source image"}),r.jsxs("span",{style:{color:"rgba(255,255,255,0.5)"},children:[g+1," / ",ie.length]}),r.jsxs("div",{style:{display:"flex",gap:"8px"},children:[r.jsx("button",{className:"overlay-btn",onClick:w=>re(xe.filename,w),title:P.has(xe.filename)?"Remove from favorites":"Add to favorites",style:{background:P.has(xe.filename)?"rgba(239, 68, 68, 0.5)":void 0},children:r.jsx(dl,{size:16,fill:P.has(xe.filename)?"#ef4444":"none",color:P.has(xe.filename)?"#ef4444":"#fff"})}),xe.has_source_image&&xe.source_image&&r.jsx("button",{className:"overlay-btn",onClick:w=>et(xe.source_image,w),title:"Download source image",children:r.jsx(mr,{size:16})}),((hn=xe.metadata)==null?void 0:hn.has_metadata)&&r.jsx("button",{className:"overlay-btn",onClick:w=>yr(xe,w),title:"Download metadata JSON",children:r.jsx(vc,{size:16})}),r.jsx("button",{className:"overlay-btn",onClick:w=>et(xe,w),title:"Download",children:r.jsx(fr,{size:16})})]})]})]}),j&&r.jsx("div",{className:"prompt-popup-overlay",onClick:()=>_(null),children:r.jsxs("div",{className:"prompt-popup",onClick:w=>w.stopPropagation(),children:[r.jsxs("div",{className:"prompt-popup-header",children:[r.jsxs("div",{className:"prompt-popup-title",children:[r.jsx(Ch,{size:18}),"Prompt Details"]}),r.jsx("button",{className:"prompt-popup-close",onClick:()=>_(null),children:r.jsx(It,{size:20})})]}),r.jsxs("div",{className:"prompt-popup-content",children:[r.jsxs("div",{style:{display:"flex",gap:"12px",alignItems:"flex-start"},children:[j.item.type==="video"?r.jsx("video",{src:`${ee}${j.item.url}`,className:"prompt-media-preview",autoPlay:!0,loop:!0,muted:!0,playsInline:!0}):r.jsx("img",{src:`${ee}${j.item.url}`,alt:j.item.filename,className:"prompt-media-preview"}),r.jsxs("div",{style:{flex:1},children:[r.jsx("div",{style:{fontSize:"0.85rem",fontWeight:600,color:"var(--text-primary)"},children:j.item.filename}),r.jsxs("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"4px"},children:[j.item.type==="video"?"🎬 Video":"🖼️ Image"," • ",Ct(j.item.size),j.item.type==="video"&&V[j.item.filename]&&r.jsxs(r.Fragment,{children:[" • ",Ec(V[j.item.filename])]}),((ms=j.item.metadata)==null?void 0:ms.width)&&((Fa=j.item.metadata)==null?void 0:Fa.height)&&r.jsxs(r.Fragment,{children:[" • ",j.item.metadata.width,"×",j.item.metadata.height]})]})]})]}),(((hs=j.item.metadata)==null?void 0:hs.positive_prompt)||((jr=j.item.metadata)==null?void 0:jr.prompt))&&r.jsxs("div",{className:"prompt-section",children:[r.jsx("div",{className:"prompt-section-label",children:"✨ Positive Prompt"}),r.jsx("div",{className:"prompt-section-text",children:j.item.metadata.positive_prompt||j.item.metadata.prompt}),r.jsxs("button",{className:"prompt-copy-btn",onClick:()=>{const w=j.item.metadata.positive_prompt||j.item.metadata.prompt;navigator.clipboard.writeText(w)},children:[r.jsx(At,{size:14}),"Copy Prompt"]})]}),((xn=j.item.metadata)==null?void 0:xn.negative_prompt)&&r.jsxs("div",{className:"prompt-section",children:[r.jsx("div",{className:"prompt-section-label",children:"🚫 Negative Prompt"}),r.jsx("div",{className:"prompt-section-text",style:{color:"var(--text-muted)"},children:j.item.metadata.negative_prompt})]}),(((gn=j.item.metadata)==null?void 0:gn.steps)||((xt=j.item.metadata)==null?void 0:xt.cfg)||((Da=j.item.metadata)==null?void 0:Da.seed)||((Fr=j.item.metadata)==null?void 0:Fr.sampler)||((xs=j.item.metadata)==null?void 0:xs.model))&&r.jsxs("div",{className:"prompt-section",children:[r.jsx("div",{className:"prompt-section-label",children:"⚙️ Generation Settings"}),r.jsxs("div",{style:{display:"flex",gap:"12px",flexWrap:"wrap",fontSize:"0.85rem"},children:[j.item.metadata.steps&&r.jsxs("span",{children:["Steps: ",r.jsx("strong",{children:j.item.metadata.steps})]}),j.item.metadata.cfg&&r.jsxs("span",{children:["CFG: ",r.jsx("strong",{children:j.item.metadata.cfg})]}),j.item.metadata.seed&&r.jsxs("span",{children:["Seed: ",r.jsx("strong",{children:j.item.metadata.seed})]}),j.item.metadata.sampler&&r.jsxs("span",{children:["Sampler: ",r.jsx("strong",{children:j.item.metadata.sampler})]}),j.item.metadata.scheduler&&r.jsxs("span",{children:["Scheduler: ",r.jsx("strong",{children:j.item.metadata.scheduler})]})]}),j.item.metadata.model&&r.jsxs("div",{style:{marginTop:"8px",fontSize:"0.8rem",color:"var(--text-muted)"},children:["Model: ",r.jsx("strong",{style:{color:"var(--text-primary)"},children:j.item.metadata.model})]})]}),((vn=j.item.metadata)==null?void 0:vn.loras)&&j.item.metadata.loras.length>0&&r.jsxs("div",{className:"prompt-section",children:[r.jsx("div",{className:"prompt-section-label",children:"🎨 LoRAs Used"}),r.jsx("div",{style:{display:"flex",flexDirection:"column",gap:"6px",fontSize:"0.85rem"},children:j.item.metadata.loras.map((w,W)=>r.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"6px 10px",backgroundColor:"var(--bg-secondary)",borderRadius:"4px"},children:[r.jsx("span",{style:{fontFamily:"monospace",fontSize:"0.8rem",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:"80%"},children:w.name}),r.jsxs("span",{style:{color:"var(--accent-color, #a855f7)",fontWeight:600,fontSize:"0.8rem"},children:[(w.strength*100).toFixed(0),"%"]})]},W))})]})]})]})})]})}const sg=()=>{const e=ee.startsWith("https")?"wss:":"ws:",t=ee.replace(/^https?:\/\//,"");return`${e}//${t}/ws/logs`};function ag(){const[e,t]=c.useState([]),[n,s]=c.useState(!0),[a,l]=c.useState(!1),[o,i]=c.useState(!1),d=c.useRef(null),u=c.useRef(null),y=c.useRef(null),g=c.useCallback(()=>{var k;if(((k=u.current)==null?void 0:k.readyState)===WebSocket.OPEN)return;const x=new WebSocket(sg());u.current=x,x.onopen=()=>{i(!0),console.log("📡 Log WebSocket connected")},x.onmessage=S=>{try{const z=JSON.parse(S.data);t(R=>[...R,z].slice(-500))}catch(z){console.error("Failed to parse log",z)}},x.onclose=()=>{i(!1),console.log("📡 Log WebSocket disconnected"),y.current=setTimeout(()=>{n&&g()},3e3)},x.onerror=S=>{console.error("WebSocket error",S),x.close()}},[n]);return c.useEffect(()=>{var x;return n?g():((x=u.current)==null||x.close(),y.current&&clearTimeout(y.current)),()=>{var k;(k=u.current)==null||k.close(),y.current&&clearTimeout(y.current)}},[n,g]),c.useEffect(()=>{d.current&&d.current.scrollIntoView({behavior:"smooth"})},[e]),n?r.jsxs("div",{style:{position:"fixed",bottom:"20px",right:"20px",width:a?"800px":"400px",height:a?"600px":"300px",backgroundColor:"#0a0a0a",border:"1px solid #333",borderRadius:"8px",display:"flex",flexDirection:"column",zIndex:100,boxShadow:"0 10px 30px rgba(0,0,0,0.8)",transition:"all 0.2s ease"},children:[r.jsxs("div",{style:{padding:"8px 12px",borderBottom:"1px solid #333",display:"flex",justifyContent:"space-between",alignItems:"center",backgroundColor:"#121212",borderTopLeftRadius:"8px",borderTopRightRadius:"8px"},children:[r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px",fontSize:"0.8rem",fontWeight:600,color:"#a3a3a3"},children:[r.jsx(jc,{size:14}),r.jsx("span",{children:"Server Logs"}),o?r.jsx(ax,{size:12,color:"#22c55e",title:"Connected"}):r.jsx(nx,{size:12,color:"#ef4444",title:"Disconnected"})]}),r.jsxs("div",{style:{display:"flex",gap:"8px"},children:[r.jsx("button",{onClick:()=>l(!a),style:{background:"transparent",border:"none",cursor:"pointer",color:"#666"},children:a?r.jsx(Th,{size:14}):r.jsx(Xu,{size:14})}),r.jsx("button",{onClick:()=>s(!1),style:{background:"transparent",border:"none",cursor:"pointer",color:"#666"},children:r.jsx(It,{size:14})})]})]}),r.jsxs("div",{style:{flex:1,overflowY:"auto",padding:"12px",fontFamily:"monospace",fontSize:"0.75rem",color:"#d4d4d4",lineHeight:"1.4"},children:[e.map((x,k)=>{var S,z;return r.jsxs("div",{style:{marginBottom:"4px",display:"flex",gap:"8px"},children:[r.jsx("span",{style:{color:"#525252",flexShrink:0},children:((z=(S=x.timestamp)==null?void 0:S.split("T")[1])==null?void 0:z.split(".")[0])||""}),r.jsx("span",{style:{color:x.level==="ERROR"?"#ef4444":x.level==="WARNING"?"#eab308":"#a3a3a3"},children:x.message})]},k)}),r.jsx("div",{ref:d})]})]}):r.jsx("button",{onClick:()=>s(!0),style:{position:"fixed",bottom:"20px",right:"20px",backgroundColor:"#1a1a1a",border:"1px solid #333",borderRadius:"50%",width:"48px",height:"48px",display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",zIndex:100,boxShadow:"0 4px 12px rgba(0,0,0,0.5)"},children:r.jsx(jc,{size:20,color:"#a3a3a3"})})}function lg(){const[e,t]=c.useState(ne.IMAGE_TO_VIDEO),[n,s]=c.useState(!1),[a,l]=c.useState(null),[o,i]=c.useState(!1),[d,u]=c.useState(null),[y,g]=c.useState(0),[x,k]=c.useState(0),[S,z]=c.useState(!1),[R,f]=c.useState(null),p=c.useRef(null),m=async()=>{try{const G=await(await fetch(`${ee}/health`)).json();l(G)}catch{l(null)}};c.useEffect(()=>{m();const I=setInterval(m,1e4);return()=>clearInterval(I)},[]);const h=async()=>{if(!o&&window.confirm("Backend herstarten? Lopende jobs worden afgebroken.")){i(!0);try{await fetch(`${ee}/restart`,{method:"POST"}),await new Promise(I=>setTimeout(I,3e3)),await m()}catch(I){console.error("Restart failed:",I)}finally{i(!1)}}},j=()=>{const I=p.current;if(!I){alert("Geen parameters beschikbaar");return}const G=new Blob([JSON.stringify(I,null,2)],{type:"application/json"}),H=URL.createObjectURL(G),N=document.createElement("a");N.href=H,N.download=`${e}_params_${Date.now()}.json`,N.click(),URL.revokeObjectURL(H)},_=c.useMemo(()=>{switch(e){case ne.TEXT_TO_VIDEO:return"Text to Video";case ne.IMAGE_TO_VIDEO:return"Image to Video";case ne.TEXT_TO_IMAGE_TO_VIDEO:return"Text to Image to Video";case ne.VIDEO_TO_VIDEO:return"Video to Video";case ne.VIDEO_TO_TEXT:return"Video to Text";case ne.PIPELINE:return"Pipeline";case ne.LORA_TRAINING:return"LoRA Training";case ne.TEXT_TO_IMAGE:return"Text to Image";case ne.IMAGE_TO_IMAGE:return"Image to Image";case ne.REFRAME:return"Reframe";case ne.FACE_SWAP:return"Face Swap";case ne.UPSCALER:return"Upscaler";case ne.IMAGE_TO_TEXT:return"Image to Text";case ne.PROMPT_GENERATOR:return"Prompt Generator";case ne.AUDIO_GENERATION:return"Audio Generation";case ne.MY_MEDIA_ALL:return"My Media - All";case ne.MY_MEDIA_VIDEOS:return"My Media - Videos";case ne.MY_MEDIA_IMAGES:return"My Media - Images";case ne.MY_MEDIA_PROMPTS:return"My Media - Prompts";default:return"Tool"}},[e]),P=()=>{const I=()=>g(C=>C+1),G=(C,L)=>{z(C),f(()=>L)},H=C=>{p.current=C},N=()=>{k(C=>C+1)};switch(e){case ne.TEXT_TO_VIDEO:return r.jsx(jx,{onOutput:u,onRefreshHistory:I,onParamsChange:H});case ne.IMAGE_TO_VIDEO:return r.jsx(_x,{onOutput:u,onRefreshHistory:I,onCreationsModeChange:G,onParamsChange:H,onJobSubmitted:N});case ne.TEXT_TO_IMAGE_TO_VIDEO:return r.jsx(zx,{onOutput:u,onParamsChange:H});case ne.PIPELINE:return r.jsx(Fx,{});case ne.LORA_TRAINING:return r.jsx(Dx,{onOutput:u});case ne.MY_MEDIA_ALL:return r.jsx(zn,{filter:"all"});case ne.MY_MEDIA_VIDEOS:return r.jsx(zn,{filter:"video"});case ne.MY_MEDIA_IMAGES:return r.jsx(zn,{filter:"image"});case ne.MY_MEDIA_PROMPTS:return r.jsx(zn,{filter:"prompts"});case ne.TEXT_TO_IMAGE:return r.jsx(Ex,{onOutput:u});case ne.IMAGE_TO_TEXT:return r.jsx($x,{});case ne.PROMPT_GENERATOR:return r.jsx(Ux,{});case ne.IMAGE_TO_IMAGE:return r.jsx(Bx,{onOutput:u});case ne.UPSCALER:return r.jsx(Gx,{onOutput:u});case ne.VIDEO_TO_VIDEO:return r.jsx(Ix,{onOutput:u});case ne.VIDEO_TO_TEXT:return r.jsx(Lx,{});case ne.AUDIO_GENERATION:return r.jsx(Yx,{onOutput:u});case ne.REFRAME:return r.jsx(qx,{onOutput:u});case ne.FACE_SWAP:return r.jsx(Zx,{onOutput:u});default:return r.jsx(eg,{title:_})}};return r.jsxs("div",{className:"dashboard-container",children:[r.jsx(px,{activeToolId:e,onSelectTool:t,collapsed:n,onToggleCollapsed:()=>s(I=>!I)}),r.jsxs("main",{className:"main-content",children:[r.jsxs("div",{className:"top-bar",children:[r.jsx("h1",{children:_}),r.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"12px"},children:[r.jsx(hx,{refreshToken:x,onJobComplete:I=>{g(G=>G+1),I.output_video&&u({kind:"video",url:`${ee}${I.output_video}`,backendUrl:`${ee}${I.output_video}`})}}),r.jsx("button",{className:"icon-btn",onClick:h,disabled:o,title:"Herstart Backend",style:{opacity:o?.5:1},children:r.jsx(dn,{size:18,color:"#fbbf24",className:o?"spin":""})}),r.jsxs("div",{className:"status-indicator",children:[r.jsx("div",{className:`status-dot ${(a==null?void 0:a.status)==="healthy"?"connected":""}`}),r.jsx("span",{children:(a==null?void 0:a.status)==="healthy"?"Connected":"Disconnected"})]})]})]}),e===ne.MY_MEDIA_ALL||e===ne.MY_MEDIA_VIDEOS||e===ne.MY_MEDIA_IMAGES||e===ne.MY_MEDIA_PROMPTS?r.jsx("div",{style:{flex:1,display:"flex",flexDirection:"column",overflow:"hidden"},children:P()}):r.jsxs("div",{className:"workspace",children:[r.jsxs("section",{className:"controls-panel",children:[r.jsxs("div",{className:"panel-header",style:{marginBottom:"16px",display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsx("div",{className:"panel-title",style:{fontSize:"0.85rem",fontWeight:600,color:"var(--text-secondary)",textTransform:"uppercase",letterSpacing:"0.05em"},children:"Parameters"}),r.jsx("button",{className:"icon-btn",onClick:j,title:"Download parameters als JSON",style:{padding:"4px"},children:r.jsx(fr,{size:16})})]}),r.jsx("div",{className:"panel-body",children:P()})]}),d?r.jsx(mx,{output:d,refreshToken:y,onSelectHistoryVideo:u,onClose:()=>u(null)}):r.jsxs("section",{className:"output-panel",style:{display:"flex",flexDirection:"column"},children:[S&&r.jsxs("div",{style:{padding:"12px 16px",borderBottom:"1px solid var(--border-color)",backgroundColor:"var(--bg-secondary)",display:"flex",justifyContent:"space-between",alignItems:"center"},children:[r.jsx("span",{style:{fontWeight:600,color:"var(--text-primary)"},children:"Select Image for I2V"}),r.jsx("span",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:"Click an image to use it"})]}),r.jsx("div",{style:{flex:1,overflow:"hidden"},children:r.jsx(zn,{filter:"all",selectionMode:S,onSelectItem:R})})]})]})]}),r.jsx(ag,{})]})}function og(){return r.jsx(lg,{})}fl.createRoot(document.getElementById("root")).render(r.jsx(yp.StrictMode,{children:r.jsx(og,{})}));
