(function(){const x=document.createElement("link").relList;if(x&&x.supports&&x.supports("modulepreload"))return;for(const p of document.querySelectorAll('link[rel="modulepreload"]'))N(p);new MutationObserver(p=>{for(const b of p)if(b.type==="childList")for(const S of b.addedNodes)S.tagName==="LINK"&&S.rel==="modulepreload"&&N(S)}).observe(document,{childList:!0,subtree:!0});function d(p){const b={};return p.integrity&&(b.integrity=p.integrity),p.referrerPolicy&&(b.referrerPolicy=p.referrerPolicy),p.crossOrigin==="use-credentials"?b.credentials="include":p.crossOrigin==="anonymous"?b.credentials="omit":b.credentials="same-origin",b}function N(p){if(p.ep)return;p.ep=!0;const b=d(p);fetch(p.href,b)}})();function iu(c){return c&&c.__esModule&&Object.prototype.hasOwnProperty.call(c,"default")?c.default:c}var zl={exports:{}},Cs={},El={exports:{}},De={};var Td;function bf(){if(Td)return De;Td=1;var c=Symbol.for("react.element"),x=Symbol.for("react.portal"),d=Symbol.for("react.fragment"),N=Symbol.for("react.strict_mode"),p=Symbol.for("react.profiler"),b=Symbol.for("react.provider"),S=Symbol.for("react.context"),R=Symbol.for("react.forward_ref"),I=Symbol.for("react.suspense"),P=Symbol.for("react.memo"),T=Symbol.for("react.lazy"),A=Symbol.iterator;function C(m){return m===null||typeof m!="object"?null:(m=A&&m[A]||m["@@iterator"],typeof m=="function"?m:null)}var L={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},V=Object.assign,U={};function D(m,$,q){this.props=m,this.context=$,this.refs=U,this.updater=q||L}D.prototype.isReactComponent={},D.prototype.setState=function(m,$){if(typeof m!="object"&&typeof m!="function"&&m!=null)throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,m,$,"setState")},D.prototype.forceUpdate=function(m){this.updater.enqueueForceUpdate(this,m,"forceUpdate")};function ee(){}ee.prototype=D.prototype;function Z(m,$,q){this.props=m,this.context=$,this.refs=U,this.updater=q||L}var K=Z.prototype=new ee;K.constructor=Z,V(K,D.prototype),K.isPureReactComponent=!0;var j=Array.isArray,k=Object.prototype.hasOwnProperty,B={current:null},h={key:!0,ref:!0,__self:!0,__source:!0};function v(m,$,q){var le,F={},_=null,Y=null;if($!=null)for(le in $.ref!==void 0&&(Y=$.ref),$.key!==void 0&&(_=""+$.key),$)k.call($,le)&&!h.hasOwnProperty(le)&&(F[le]=$[le]);var Q=arguments.length-2;if(Q===1)F.children=q;else if(1<Q){for(var u=Array(Q),he=0;he<Q;he++)u[he]=arguments[he+2];F.children=u}if(m&&m.defaultProps)for(le in Q=m.defaultProps,Q)F[le]===void 0&&(F[le]=Q[le]);return{$$typeof:c,type:m,key:_,ref:Y,props:F,_owner:B.current}}function te(m,$){return{$$typeof:c,type:m.type,key:$,ref:m.ref,props:m.props,_owner:m._owner}}function re(m){return typeof m=="object"&&m!==null&&m.$$typeof===c}function xe(m){var $={"=":"=0",":":"=2"};return"$"+m.replace(/[=:]/g,function(q){return $[q]})}var ge=/\/+/g;function E(m,$){return typeof m=="object"&&m!==null&&m.key!=null?xe(""+m.key):$.toString(36)}function ue(m,$,q,le,F){var _=typeof m;(_==="undefined"||_==="boolean")&&(m=null);var Y=!1;if(m===null)Y=!0;else switch(_){case"string":case"number":Y=!0;break;case"object":switch(m.$$typeof){case c:case x:Y=!0}}if(Y)return Y=m,F=F(Y),m=le===""?"."+E(Y,0):le,j(F)?(q="",m!=null&&(q=m.replace(ge,"$&/")+"/"),ue(F,$,q,"",function(he){return he})):F!=null&&(re(F)&&(F=te(F,q+(!F.key||Y&&Y.key===F.key?"":(""+F.key).replace(ge,"$&/")+"/")+m)),$.push(F)),1;if(Y=0,le=le===""?".":le+":",j(m))for(var Q=0;Q<m.length;Q++){_=m[Q];var u=le+E(_,Q);Y+=ue(_,$,q,u,F)}else if(u=C(m),typeof u=="function")for(m=u.call(m),Q=0;!(_=m.next()).done;)_=_.value,u=le+E(_,Q++),Y+=ue(_,$,q,u,F);else if(_==="object")throw $=String(m),Error("Objects are not valid as a React child (found: "+($==="[object Object]"?"object with keys {"+Object.keys(m).join(", ")+"}":$)+"). If you meant to render a collection of children, use an array instead.");return Y}function fe(m,$,q){if(m==null)return m;var le=[],F=0;return ue(m,le,"","",function(_){return $.call(q,_,F++)}),le}function ie(m){if(m._status===-1){var $=m._result;$=$(),$.then(function(q){(m._status===0||m._status===-1)&&(m._status=1,m._result=q)},function(q){(m._status===0||m._status===-1)&&(m._status=2,m._result=q)}),m._status===-1&&(m._status=0,m._result=$)}if(m._status===1)return m._result.default;throw m._result}var W={current:null},G={transition:null},X={ReactCurrentDispatcher:W,ReactCurrentBatchConfig:G,ReactCurrentOwner:B};function J(){throw Error("act(...) is not supported in production builds of React.")}return De.Children={map:fe,forEach:function(m,$,q){fe(m,function(){$.apply(this,arguments)},q)},count:function(m){var $=0;return fe(m,function(){$++}),$},toArray:function(m){return fe(m,function($){return $})||[]},only:function(m){if(!re(m))throw Error("React.Children.only expected to receive a single React element child.");return m}},De.Component=D,De.Fragment=d,De.Profiler=p,De.PureComponent=Z,De.StrictMode=N,De.Suspense=I,De.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=X,De.act=J,De.cloneElement=function(m,$,q){if(m==null)throw Error("React.cloneElement(...): The argument must be a React element, but you passed "+m+".");var le=V({},m.props),F=m.key,_=m.ref,Y=m._owner;if($!=null){if($.ref!==void 0&&(_=$.ref,Y=B.current),$.key!==void 0&&(F=""+$.key),m.type&&m.type.defaultProps)var Q=m.type.defaultProps;for(u in $)k.call($,u)&&!h.hasOwnProperty(u)&&(le[u]=$[u]===void 0&&Q!==void 0?Q[u]:$[u])}var u=arguments.length-2;if(u===1)le.children=q;else if(1<u){Q=Array(u);for(var he=0;he<u;he++)Q[he]=arguments[he+2];le.children=Q}return{$$typeof:c,type:m.type,key:F,ref:_,props:le,_owner:Y}},De.createContext=function(m){return m={$$typeof:S,_currentValue:m,_currentValue2:m,_threadCount:0,Provider:null,Consumer:null,_defaultValue:null,_globalName:null},m.Provider={$$typeof:b,_context:m},m.Consumer=m},De.createElement=v,De.createFactory=function(m){var $=v.bind(null,m);return $.type=m,$},De.createRef=function(){return{current:null}},De.forwardRef=function(m){return{$$typeof:R,render:m}},De.isValidElement=re,De.lazy=function(m){return{$$typeof:T,_payload:{_status:-1,_result:m},_init:ie}},De.memo=function(m,$){return{$$typeof:P,type:m,compare:$===void 0?null:$}},De.startTransition=function(m){var $=G.transition;G.transition={};try{m()}finally{G.transition=$}},De.unstable_act=J,De.useCallback=function(m,$){return W.current.useCallback(m,$)},De.useContext=function(m){return W.current.useContext(m)},De.useDebugValue=function(){},De.useDeferredValue=function(m){return W.current.useDeferredValue(m)},De.useEffect=function(m,$){return W.current.useEffect(m,$)},De.useId=function(){return W.current.useId()},De.useImperativeHandle=function(m,$,q){return W.current.useImperativeHandle(m,$,q)},De.useInsertionEffect=function(m,$){return W.current.useInsertionEffect(m,$)},De.useLayoutEffect=function(m,$){return W.current.useLayoutEffect(m,$)},De.useMemo=function(m,$){return W.current.useMemo(m,$)},De.useReducer=function(m,$,q){return W.current.useReducer(m,$,q)},De.useRef=function(m){return W.current.useRef(m)},De.useState=function(m){return W.current.useState(m)},De.useSyncExternalStore=function(m,$,q){return W.current.useSyncExternalStore(m,$,q)},De.useTransition=function(){return W.current.useTransition()},De.version="18.3.1",De}var Rd;function Ql(){return Rd||(Rd=1,El.exports=bf()),El.exports}var Md;function jf(){if(Md)return Cs;Md=1;var c=Ql(),x=Symbol.for("react.element"),d=Symbol.for("react.fragment"),N=Object.prototype.hasOwnProperty,p=c.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,b={key:!0,ref:!0,__self:!0,__source:!0};function S(R,I,P){var T,A={},C=null,L=null;P!==void 0&&(C=""+P),I.key!==void 0&&(C=""+I.key),I.ref!==void 0&&(L=I.ref);for(T in I)N.call(I,T)&&!b.hasOwnProperty(T)&&(A[T]=I[T]);if(R&&R.defaultProps)for(T in I=R.defaultProps,I)A[T]===void 0&&(A[T]=I[T]);return{$$typeof:x,type:R,key:C,ref:L,props:A,_owner:p.current}}return Cs.Fragment=d,Cs.jsx=S,Cs.jsxs=S,Cs}var Ld;function wf(){return Ld||(Ld=1,zl.exports=jf()),zl.exports}var t=wf(),l=Ql();const kf=iu(l);var Oa={},Il={exports:{}},Rt={},Pl={exports:{}},Tl={};var Fd;function Sf(){return Fd||(Fd=1,(function(c){function x(G,X){var J=G.length;G.push(X);e:for(;0<J;){var m=J-1>>>1,$=G[m];if(0<p($,X))G[m]=X,G[J]=$,J=m;else break e}}function d(G){return G.length===0?null:G[0]}function N(G){if(G.length===0)return null;var X=G[0],J=G.pop();if(J!==X){G[0]=J;e:for(var m=0,$=G.length,q=$>>>1;m<q;){var le=2*(m+1)-1,F=G[le],_=le+1,Y=G[_];if(0>p(F,J))_<$&&0>p(Y,F)?(G[m]=Y,G[_]=J,m=_):(G[m]=F,G[le]=J,m=le);else if(_<$&&0>p(Y,J))G[m]=Y,G[_]=J,m=_;else break e}}return X}function p(G,X){var J=G.sortIndex-X.sortIndex;return J!==0?J:G.id-X.id}if(typeof performance=="object"&&typeof performance.now=="function"){var b=performance;c.unstable_now=function(){return b.now()}}else{var S=Date,R=S.now();c.unstable_now=function(){return S.now()-R}}var I=[],P=[],T=1,A=null,C=3,L=!1,V=!1,U=!1,D=typeof setTimeout=="function"?setTimeout:null,ee=typeof clearTimeout=="function"?clearTimeout:null,Z=typeof setImmediate<"u"?setImmediate:null;typeof navigator<"u"&&navigator.scheduling!==void 0&&navigator.scheduling.isInputPending!==void 0&&navigator.scheduling.isInputPending.bind(navigator.scheduling);function K(G){for(var X=d(P);X!==null;){if(X.callback===null)N(P);else if(X.startTime<=G)N(P),X.sortIndex=X.expirationTime,x(I,X);else break;X=d(P)}}function j(G){if(U=!1,K(G),!V)if(d(I)!==null)V=!0,ie(k);else{var X=d(P);X!==null&&W(j,X.startTime-G)}}function k(G,X){V=!1,U&&(U=!1,ee(v),v=-1),L=!0;var J=C;try{for(K(X),A=d(I);A!==null&&(!(A.expirationTime>X)||G&&!xe());){var m=A.callback;if(typeof m=="function"){A.callback=null,C=A.priorityLevel;var $=m(A.expirationTime<=X);X=c.unstable_now(),typeof $=="function"?A.callback=$:A===d(I)&&N(I),K(X)}else N(I);A=d(I)}if(A!==null)var q=!0;else{var le=d(P);le!==null&&W(j,le.startTime-X),q=!1}return q}finally{A=null,C=J,L=!1}}var B=!1,h=null,v=-1,te=5,re=-1;function xe(){return!(c.unstable_now()-re<te)}function ge(){if(h!==null){var G=c.unstable_now();re=G;var X=!0;try{X=h(!0,G)}finally{X?E():(B=!1,h=null)}}else B=!1}var E;if(typeof Z=="function")E=function(){Z(ge)};else if(typeof MessageChannel<"u"){var ue=new MessageChannel,fe=ue.port2;ue.port1.onmessage=ge,E=function(){fe.postMessage(null)}}else E=function(){D(ge,0)};function ie(G){h=G,B||(B=!0,E())}function W(G,X){v=D(function(){G(c.unstable_now())},X)}c.unstable_IdlePriority=5,c.unstable_ImmediatePriority=1,c.unstable_LowPriority=4,c.unstable_NormalPriority=3,c.unstable_Profiling=null,c.unstable_UserBlockingPriority=2,c.unstable_cancelCallback=function(G){G.callback=null},c.unstable_continueExecution=function(){V||L||(V=!0,ie(k))},c.unstable_forceFrameRate=function(G){0>G||125<G?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):te=0<G?Math.floor(1e3/G):5},c.unstable_getCurrentPriorityLevel=function(){return C},c.unstable_getFirstCallbackNode=function(){return d(I)},c.unstable_next=function(G){switch(C){case 1:case 2:case 3:var X=3;break;default:X=C}var J=C;C=X;try{return G()}finally{C=J}},c.unstable_pauseExecution=function(){},c.unstable_requestPaint=function(){},c.unstable_runWithPriority=function(G,X){switch(G){case 1:case 2:case 3:case 4:case 5:break;default:G=3}var J=C;C=G;try{return X()}finally{C=J}},c.unstable_scheduleCallback=function(G,X,J){var m=c.unstable_now();switch(typeof J=="object"&&J!==null?(J=J.delay,J=typeof J=="number"&&0<J?m+J:m):J=m,G){case 1:var $=-1;break;case 2:$=250;break;case 5:$=1073741823;break;case 4:$=1e4;break;default:$=5e3}return $=J+$,G={id:T++,callback:X,priorityLevel:G,startTime:J,expirationTime:$,sortIndex:-1},J>m?(G.sortIndex=J,x(P,G),d(I)===null&&G===d(P)&&(U?(ee(v),v=-1):U=!0,W(j,J-m))):(G.sortIndex=$,x(I,G),V||L||(V=!0,ie(k))),G},c.unstable_shouldYield=xe,c.unstable_wrapCallback=function(G){var X=C;return function(){var J=C;C=X;try{return G.apply(this,arguments)}finally{C=J}}}})(Tl)),Tl}var Dd;function Nf(){return Dd||(Dd=1,Pl.exports=Sf()),Pl.exports}var Od;function Cf(){if(Od)return Rt;Od=1;var c=Ql(),x=Nf();function d(e){for(var n="https://reactjs.org/docs/error-decoder.html?invariant="+e,r=1;r<arguments.length;r++)n+="&args[]="+encodeURIComponent(arguments[r]);return"Minified React error #"+e+"; visit "+n+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}var N=new Set,p={};function b(e,n){S(e,n),S(e+"Capture",n)}function S(e,n){for(p[e]=n,e=0;e<n.length;e++)N.add(n[e])}var R=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),I=Object.prototype.hasOwnProperty,P=/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,T={},A={};function C(e){return I.call(A,e)?!0:I.call(T,e)?!1:P.test(e)?A[e]=!0:(T[e]=!0,!1)}function L(e,n,r,s){if(r!==null&&r.type===0)return!1;switch(typeof n){case"function":case"symbol":return!0;case"boolean":return s?!1:r!==null?!r.acceptsBooleans:(e=e.toLowerCase().slice(0,5),e!=="data-"&&e!=="aria-");default:return!1}}function V(e,n,r,s){if(n===null||typeof n>"u"||L(e,n,r,s))return!0;if(s)return!1;if(r!==null)switch(r.type){case 3:return!n;case 4:return n===!1;case 5:return isNaN(n);case 6:return isNaN(n)||1>n}return!1}function U(e,n,r,s,a,o,i){this.acceptsBooleans=n===2||n===3||n===4,this.attributeName=s,this.attributeNamespace=a,this.mustUseProperty=r,this.propertyName=e,this.type=n,this.sanitizeURL=o,this.removeEmptyString=i}var D={};"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e){D[e]=new U(e,0,!1,e,null,!1,!1)}),[["acceptCharset","accept-charset"],["className","class"],["htmlFor","for"],["httpEquiv","http-equiv"]].forEach(function(e){var n=e[0];D[n]=new U(n,1,!1,e[1],null,!1,!1)}),["contentEditable","draggable","spellCheck","value"].forEach(function(e){D[e]=new U(e,2,!1,e.toLowerCase(),null,!1,!1)}),["autoReverse","externalResourcesRequired","focusable","preserveAlpha"].forEach(function(e){D[e]=new U(e,2,!1,e,null,!1,!1)}),"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e){D[e]=new U(e,3,!1,e.toLowerCase(),null,!1,!1)}),["checked","multiple","muted","selected"].forEach(function(e){D[e]=new U(e,3,!0,e,null,!1,!1)}),["capture","download"].forEach(function(e){D[e]=new U(e,4,!1,e,null,!1,!1)}),["cols","rows","size","span"].forEach(function(e){D[e]=new U(e,6,!1,e,null,!1,!1)}),["rowSpan","start"].forEach(function(e){D[e]=new U(e,5,!1,e.toLowerCase(),null,!1,!1)});var ee=/[\-:]([a-z])/g;function Z(e){return e[1].toUpperCase()}"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e){var n=e.replace(ee,Z);D[n]=new U(n,1,!1,e,null,!1,!1)}),"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e){var n=e.replace(ee,Z);D[n]=new U(n,1,!1,e,"http://www.w3.org/1999/xlink",!1,!1)}),["xml:base","xml:lang","xml:space"].forEach(function(e){var n=e.replace(ee,Z);D[n]=new U(n,1,!1,e,"http://www.w3.org/XML/1998/namespace",!1,!1)}),["tabIndex","crossOrigin"].forEach(function(e){D[e]=new U(e,1,!1,e.toLowerCase(),null,!1,!1)}),D.xlinkHref=new U("xlinkHref",1,!1,"xlink:href","http://www.w3.org/1999/xlink",!0,!1),["src","href","action","formAction"].forEach(function(e){D[e]=new U(e,1,!1,e.toLowerCase(),null,!0,!0)});function K(e,n,r,s){var a=D.hasOwnProperty(n)?D[n]:null;(a!==null?a.type!==0:s||!(2<n.length)||n[0]!=="o"&&n[0]!=="O"||n[1]!=="n"&&n[1]!=="N")&&(V(n,r,a,s)&&(r=null),s||a===null?C(n)&&(r===null?e.removeAttribute(n):e.setAttribute(n,""+r)):a.mustUseProperty?e[a.propertyName]=r===null?a.type===3?!1:"":r:(n=a.attributeName,s=a.attributeNamespace,r===null?e.removeAttribute(n):(a=a.type,r=a===3||a===4&&r===!0?"":""+r,s?e.setAttributeNS(s,n,r):e.setAttribute(n,r))))}var j=c.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,k=Symbol.for("react.element"),B=Symbol.for("react.portal"),h=Symbol.for("react.fragment"),v=Symbol.for("react.strict_mode"),te=Symbol.for("react.profiler"),re=Symbol.for("react.provider"),xe=Symbol.for("react.context"),ge=Symbol.for("react.forward_ref"),E=Symbol.for("react.suspense"),ue=Symbol.for("react.suspense_list"),fe=Symbol.for("react.memo"),ie=Symbol.for("react.lazy"),W=Symbol.for("react.offscreen"),G=Symbol.iterator;function X(e){return e===null||typeof e!="object"?null:(e=G&&e[G]||e["@@iterator"],typeof e=="function"?e:null)}var J=Object.assign,m;function $(e){if(m===void 0)try{throw Error()}catch(r){var n=r.stack.trim().match(/\n( *(at )?)/);m=n&&n[1]||""}return`
`+m+e}var q=!1;function le(e,n){if(!e||q)return"";q=!0;var r=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{if(n)if(n=function(){throw Error()},Object.defineProperty(n.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(n,[])}catch(O){var s=O}Reflect.construct(e,[],n)}else{try{n.call()}catch(O){s=O}e.call(n.prototype)}else{try{throw Error()}catch(O){s=O}e()}}catch(O){if(O&&s&&typeof O.stack=="string"){for(var a=O.stack.split(`
`),o=s.stack.split(`
`),i=a.length-1,f=o.length-1;1<=i&&0<=f&&a[i]!==o[f];)f--;for(;1<=i&&0<=f;i--,f--)if(a[i]!==o[f]){if(i!==1||f!==1)do if(i--,f--,0>f||a[i]!==o[f]){var g=`
`+a[i].replace(" at new "," at ");return e.displayName&&g.includes("<anonymous>")&&(g=g.replace("<anonymous>",e.displayName)),g}while(1<=i&&0<=f);break}}}finally{q=!1,Error.prepareStackTrace=r}return(e=e?e.displayName||e.name:"")?$(e):""}function F(e){switch(e.tag){case 5:return $(e.type);case 16:return $("Lazy");case 13:return $("Suspense");case 19:return $("SuspenseList");case 0:case 2:case 15:return e=le(e.type,!1),e;case 11:return e=le(e.type.render,!1),e;case 1:return e=le(e.type,!0),e;default:return""}}function _(e){if(e==null)return null;if(typeof e=="function")return e.displayName||e.name||null;if(typeof e=="string")return e;switch(e){case h:return"Fragment";case B:return"Portal";case te:return"Profiler";case v:return"StrictMode";case E:return"Suspense";case ue:return"SuspenseList"}if(typeof e=="object")switch(e.$$typeof){case xe:return(e.displayName||"Context")+".Consumer";case re:return(e._context.displayName||"Context")+".Provider";case ge:var n=e.render;return e=e.displayName,e||(e=n.displayName||n.name||"",e=e!==""?"ForwardRef("+e+")":"ForwardRef"),e;case fe:return n=e.displayName||null,n!==null?n:_(e.type)||"Memo";case ie:n=e._payload,e=e._init;try{return _(e(n))}catch{}}return null}function Y(e){var n=e.type;switch(e.tag){case 24:return"Cache";case 9:return(n.displayName||"Context")+".Consumer";case 10:return(n._context.displayName||"Context")+".Provider";case 18:return"DehydratedFragment";case 11:return e=n.render,e=e.displayName||e.name||"",n.displayName||(e!==""?"ForwardRef("+e+")":"ForwardRef");case 7:return"Fragment";case 5:return n;case 4:return"Portal";case 3:return"Root";case 6:return"Text";case 16:return _(n);case 8:return n===v?"StrictMode":"Mode";case 22:return"Offscreen";case 12:return"Profiler";case 21:return"Scope";case 13:return"Suspense";case 19:return"SuspenseList";case 25:return"TracingMarker";case 1:case 0:case 17:case 2:case 14:case 15:if(typeof n=="function")return n.displayName||n.name||null;if(typeof n=="string")return n}return null}function Q(e){switch(typeof e){case"boolean":case"number":case"string":case"undefined":return e;case"object":return e;default:return""}}function u(e){var n=e.type;return(e=e.nodeName)&&e.toLowerCase()==="input"&&(n==="checkbox"||n==="radio")}function he(e){var n=u(e)?"checked":"value",r=Object.getOwnPropertyDescriptor(e.constructor.prototype,n),s=""+e[n];if(!e.hasOwnProperty(n)&&typeof r<"u"&&typeof r.get=="function"&&typeof r.set=="function"){var a=r.get,o=r.set;return Object.defineProperty(e,n,{configurable:!0,get:function(){return a.call(this)},set:function(i){s=""+i,o.call(this,i)}}),Object.defineProperty(e,n,{enumerable:r.enumerable}),{getValue:function(){return s},setValue:function(i){s=""+i},stopTracking:function(){e._valueTracker=null,delete e[n]}}}}function ze(e){e._valueTracker||(e._valueTracker=he(e))}function pe(e){if(!e)return!1;var n=e._valueTracker;if(!n)return!0;var r=n.getValue(),s="";return e&&(s=u(e)?e.checked?"true":"false":e.value),e=s,e!==r?(n.setValue(e),!0):!1}function Ne(e){if(e=e||(typeof document<"u"?document:void 0),typeof e>"u")return null;try{return e.activeElement||e.body}catch{return e.body}}function Re(e,n){var r=n.checked;return J({},n,{defaultChecked:void 0,defaultValue:void 0,value:void 0,checked:r??e._wrapperState.initialChecked})}function Ve(e,n){var r=n.defaultValue==null?"":n.defaultValue,s=n.checked!=null?n.checked:n.defaultChecked;r=Q(n.value!=null?n.value:r),e._wrapperState={initialChecked:s,initialValue:r,controlled:n.type==="checkbox"||n.type==="radio"?n.checked!=null:n.value!=null}}function it(e,n){n=n.checked,n!=null&&K(e,"checked",n,!1)}function Je(e,n){it(e,n);var r=Q(n.value),s=n.type;if(r!=null)s==="number"?(r===0&&e.value===""||e.value!=r)&&(e.value=""+r):e.value!==""+r&&(e.value=""+r);else if(s==="submit"||s==="reset"){e.removeAttribute("value");return}n.hasOwnProperty("value")?_t(e,n.type,r):n.hasOwnProperty("defaultValue")&&_t(e,n.type,Q(n.defaultValue)),n.checked==null&&n.defaultChecked!=null&&(e.defaultChecked=!!n.defaultChecked)}function Kn(e,n,r){if(n.hasOwnProperty("value")||n.hasOwnProperty("defaultValue")){var s=n.type;if(!(s!=="submit"&&s!=="reset"||n.value!==void 0&&n.value!==null))return;n=""+e._wrapperState.initialValue,r||n===e.value||(e.value=n),e.defaultValue=n}r=e.name,r!==""&&(e.name=""),e.defaultChecked=!!e._wrapperState.initialChecked,r!==""&&(e.name=r)}function _t(e,n,r){(n!=="number"||Ne(e.ownerDocument)!==e)&&(r==null?e.defaultValue=""+e._wrapperState.initialValue:e.defaultValue!==""+r&&(e.defaultValue=""+r))}var We=Array.isArray;function kt(e,n,r,s){if(e=e.options,n){n={};for(var a=0;a<r.length;a++)n["$"+r[a]]=!0;for(r=0;r<e.length;r++)a=n.hasOwnProperty("$"+e[r].value),e[r].selected!==a&&(e[r].selected=a),a&&s&&(e[r].defaultSelected=!0)}else{for(r=""+Q(r),n=null,a=0;a<e.length;a++){if(e[a].value===r){e[a].selected=!0,s&&(e[a].defaultSelected=!0);return}n!==null||e[a].disabled||(n=e[a])}n!==null&&(n.selected=!0)}}function _n(e,n){if(n.dangerouslySetInnerHTML!=null)throw Error(d(91));return J({},n,{value:void 0,defaultValue:void 0,children:""+e._wrapperState.initialValue})}function pn(e,n){var r=n.value;if(r==null){if(r=n.children,n=n.defaultValue,r!=null){if(n!=null)throw Error(d(92));if(We(r)){if(1<r.length)throw Error(d(93));r=r[0]}n=r}n==null&&(n=""),r=n}e._wrapperState={initialValue:Q(r)}}function hr(e,n){var r=Q(n.value),s=Q(n.defaultValue);r!=null&&(r=""+r,r!==e.value&&(e.value=r),n.defaultValue==null&&e.defaultValue!==r&&(e.defaultValue=r)),s!=null&&(e.defaultValue=""+s)}function zn(e){var n=e.textContent;n===e._wrapperState.initialValue&&n!==""&&n!==null&&(e.value=n)}function gr(e){switch(e){case"svg":return"http://www.w3.org/2000/svg";case"math":return"http://www.w3.org/1998/Math/MathML";default:return"http://www.w3.org/1999/xhtml"}}function Mt(e,n){return e==null||e==="http://www.w3.org/1999/xhtml"?gr(n):e==="http://www.w3.org/2000/svg"&&n==="foreignObject"?"http://www.w3.org/1999/xhtml":e}var fn,mn=(function(e){return typeof MSApp<"u"&&MSApp.execUnsafeLocalFunction?function(n,r,s,a){MSApp.execUnsafeLocalFunction(function(){return e(n,r,s,a)})}:e})(function(e,n){if(e.namespaceURI!=="http://www.w3.org/2000/svg"||"innerHTML"in e)e.innerHTML=n;else{for(fn=fn||document.createElement("div"),fn.innerHTML="<svg>"+n.valueOf().toString()+"</svg>",n=fn.firstChild;e.firstChild;)e.removeChild(e.firstChild);for(;n.firstChild;)e.appendChild(n.firstChild)}});function $t(e,n){if(n){var r=e.firstChild;if(r&&r===e.lastChild&&r.nodeType===3){r.nodeValue=n;return}}e.textContent=n}var Fe={animationIterationCount:!0,aspectRatio:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridArea:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},Wr=["Webkit","ms","Moz","O"];Object.keys(Fe).forEach(function(e){Wr.forEach(function(n){n=n+e.charAt(0).toUpperCase()+e.substring(1),Fe[n]=Fe[e]})});function w(e,n,r){return n==null||typeof n=="boolean"||n===""?"":r||typeof n!="number"||n===0||Fe.hasOwnProperty(e)&&Fe[e]?(""+n).trim():n+"px"}function ce(e,n){e=e.style;for(var r in n)if(n.hasOwnProperty(r)){var s=r.indexOf("--")===0,a=w(r,n[r],s);r==="float"&&(r="cssFloat"),s?e.setProperty(r,a):e[r]=a}}var ae=J({menuitem:!0},{area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0});function ke(e,n){if(n){if(ae[e]&&(n.children!=null||n.dangerouslySetInnerHTML!=null))throw Error(d(137,e));if(n.dangerouslySetInnerHTML!=null){if(n.children!=null)throw Error(d(60));if(typeof n.dangerouslySetInnerHTML!="object"||!("__html"in n.dangerouslySetInnerHTML))throw Error(d(61))}if(n.style!=null&&typeof n.style!="object")throw Error(d(62))}}function Ae(e,n){if(e.indexOf("-")===-1)return typeof n.is=="string";switch(e){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var Oe=null;function ct(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var St=null,Qe=null,Ge=null;function Lt(e){if(e=us(e)){if(typeof St!="function")throw Error(d(280));var n=e.stateNode;n&&(n=Zs(n),St(e.stateNode,e.type,n))}}function Gr(e){Qe?Ge?Ge.push(e):Ge=[e]:Qe=e}function Hr(){if(Qe){var e=Qe,n=Ge;if(Ge=Qe=null,Lt(e),n)for(e=0;e<n.length;e++)Lt(n[e])}}function Is(e,n){return e(n)}function Ps(){}var H=!1;function be(e,n,r){if(H)return e(n,r);H=!0;try{return Is(e,n,r)}finally{H=!1,(Qe!==null||Ge!==null)&&(Ps(),Hr())}}function Te(e,n){var r=e.stateNode;if(r===null)return null;var s=Zs(r);if(s===null)return null;r=s[n];e:switch(n){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(s=!s.disabled)||(e=e.type,s=!(e==="button"||e==="input"||e==="select"||e==="textarea")),e=!s;break e;default:e=!1}if(e)return null;if(r&&typeof r!="function")throw Error(d(231,n,typeof r));return r}var Le=!1;if(R)try{var He={};Object.defineProperty(He,"passive",{get:function(){Le=!0}}),window.addEventListener("test",He,He),window.removeEventListener("test",He,He)}catch{Le=!1}function Yt(e,n,r,s,a,o,i,f,g){var O=Array.prototype.slice.call(arguments,3);try{n.apply(r,O)}catch(se){this.onError(se)}}var Jn=!1,Zn=null,Ts=!1,Ha=null,Cu={onError:function(e){Jn=!0,Zn=e}};function _u(e,n,r,s,a,o,i,f,g){Jn=!1,Zn=null,Yt.apply(Cu,arguments)}function zu(e,n,r,s,a,o,i,f,g){if(_u.apply(this,arguments),Jn){if(Jn){var O=Zn;Jn=!1,Zn=null}else throw Error(d(198));Ts||(Ts=!0,Ha=O)}}function er(e){var n=e,r=e;if(e.alternate)for(;n.return;)n=n.return;else{e=n;do n=e,(n.flags&4098)!==0&&(r=n.return),e=n.return;while(e)}return n.tag===3?r:null}function ti(e){if(e.tag===13){var n=e.memoizedState;if(n===null&&(e=e.alternate,e!==null&&(n=e.memoizedState)),n!==null)return n.dehydrated}return null}function ni(e){if(er(e)!==e)throw Error(d(188))}function Eu(e){var n=e.alternate;if(!n){if(n=er(e),n===null)throw Error(d(188));return n!==e?null:e}for(var r=e,s=n;;){var a=r.return;if(a===null)break;var o=a.alternate;if(o===null){if(s=a.return,s!==null){r=s;continue}break}if(a.child===o.child){for(o=a.child;o;){if(o===r)return ni(a),e;if(o===s)return ni(a),n;o=o.sibling}throw Error(d(188))}if(r.return!==s.return)r=a,s=o;else{for(var i=!1,f=a.child;f;){if(f===r){i=!0,r=a,s=o;break}if(f===s){i=!0,s=a,r=o;break}f=f.sibling}if(!i){for(f=o.child;f;){if(f===r){i=!0,r=o,s=a;break}if(f===s){i=!0,s=o,r=a;break}f=f.sibling}if(!i)throw Error(d(189))}}if(r.alternate!==s)throw Error(d(190))}if(r.tag!==3)throw Error(d(188));return r.stateNode.current===r?e:n}function ri(e){return e=Eu(e),e!==null?si(e):null}function si(e){if(e.tag===5||e.tag===6)return e;for(e=e.child;e!==null;){var n=si(e);if(n!==null)return n;e=e.sibling}return null}var ai=x.unstable_scheduleCallback,oi=x.unstable_cancelCallback,Iu=x.unstable_shouldYield,Pu=x.unstable_requestPaint,rt=x.unstable_now,Tu=x.unstable_getCurrentPriorityLevel,qa=x.unstable_ImmediatePriority,li=x.unstable_UserBlockingPriority,Rs=x.unstable_NormalPriority,Ru=x.unstable_LowPriority,ii=x.unstable_IdlePriority,Ms=null,sn=null;function Mu(e){if(sn&&typeof sn.onCommitFiberRoot=="function")try{sn.onCommitFiberRoot(Ms,e,void 0,(e.current.flags&128)===128)}catch{}}var Xt=Math.clz32?Math.clz32:Du,Lu=Math.log,Fu=Math.LN2;function Du(e){return e>>>=0,e===0?32:31-(Lu(e)/Fu|0)|0}var Ls=64,Fs=4194304;function qr(e){switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e&4194240;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return e&130023424;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 1073741824;default:return e}}function Ds(e,n){var r=e.pendingLanes;if(r===0)return 0;var s=0,a=e.suspendedLanes,o=e.pingedLanes,i=r&268435455;if(i!==0){var f=i&~a;f!==0?s=qr(f):(o&=i,o!==0&&(s=qr(o)))}else i=r&~a,i!==0?s=qr(i):o!==0&&(s=qr(o));if(s===0)return 0;if(n!==0&&n!==s&&(n&a)===0&&(a=s&-s,o=n&-n,a>=o||a===16&&(o&4194240)!==0))return n;if((s&4)!==0&&(s|=r&16),n=e.entangledLanes,n!==0)for(e=e.entanglements,n&=s;0<n;)r=31-Xt(n),a=1<<r,s|=e[r],n&=~a;return s}function Ou(e,n){switch(e){case 1:case 2:case 4:return n+250;case 8:case 16:case 32:case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return n+5e3;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return-1;case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function Au(e,n){for(var r=e.suspendedLanes,s=e.pingedLanes,a=e.expirationTimes,o=e.pendingLanes;0<o;){var i=31-Xt(o),f=1<<i,g=a[i];g===-1?((f&r)===0||(f&s)!==0)&&(a[i]=Ou(f,n)):g<=n&&(e.expiredLanes|=f),o&=~f}}function Qa(e){return e=e.pendingLanes&-1073741825,e!==0?e:e&1073741824?1073741824:0}function ci(){var e=Ls;return Ls<<=1,(Ls&4194240)===0&&(Ls=64),e}function Ya(e){for(var n=[],r=0;31>r;r++)n.push(e);return n}function Qr(e,n,r){e.pendingLanes|=n,n!==536870912&&(e.suspendedLanes=0,e.pingedLanes=0),e=e.eventTimes,n=31-Xt(n),e[n]=r}function $u(e,n){var r=e.pendingLanes&~n;e.pendingLanes=n,e.suspendedLanes=0,e.pingedLanes=0,e.expiredLanes&=n,e.mutableReadLanes&=n,e.entangledLanes&=n,n=e.entanglements;var s=e.eventTimes;for(e=e.expirationTimes;0<r;){var a=31-Xt(r),o=1<<a;n[a]=0,s[a]=-1,e[a]=-1,r&=~o}}function Xa(e,n){var r=e.entangledLanes|=n;for(e=e.entanglements;r;){var s=31-Xt(r),a=1<<s;a&n|e[s]&n&&(e[s]|=n),r&=~a}}var Be=0;function di(e){return e&=-e,1<e?4<e?(e&268435455)!==0?16:536870912:4:1}var ui,Ka,pi,fi,mi,Ja=!1,Os=[],En=null,In=null,Pn=null,Yr=new Map,Xr=new Map,Tn=[],Uu="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");function xi(e,n){switch(e){case"focusin":case"focusout":En=null;break;case"dragenter":case"dragleave":In=null;break;case"mouseover":case"mouseout":Pn=null;break;case"pointerover":case"pointerout":Yr.delete(n.pointerId);break;case"gotpointercapture":case"lostpointercapture":Xr.delete(n.pointerId)}}function Kr(e,n,r,s,a,o){return e===null||e.nativeEvent!==o?(e={blockedOn:n,domEventName:r,eventSystemFlags:s,nativeEvent:o,targetContainers:[a]},n!==null&&(n=us(n),n!==null&&Ka(n)),e):(e.eventSystemFlags|=s,n=e.targetContainers,a!==null&&n.indexOf(a)===-1&&n.push(a),e)}function Vu(e,n,r,s,a){switch(n){case"focusin":return En=Kr(En,e,n,r,s,a),!0;case"dragenter":return In=Kr(In,e,n,r,s,a),!0;case"mouseover":return Pn=Kr(Pn,e,n,r,s,a),!0;case"pointerover":var o=a.pointerId;return Yr.set(o,Kr(Yr.get(o)||null,e,n,r,s,a)),!0;case"gotpointercapture":return o=a.pointerId,Xr.set(o,Kr(Xr.get(o)||null,e,n,r,s,a)),!0}return!1}function hi(e){var n=tr(e.target);if(n!==null){var r=er(n);if(r!==null){if(n=r.tag,n===13){if(n=ti(r),n!==null){e.blockedOn=n,mi(e.priority,function(){pi(r)});return}}else if(n===3&&r.stateNode.current.memoizedState.isDehydrated){e.blockedOn=r.tag===3?r.stateNode.containerInfo:null;return}}}e.blockedOn=null}function As(e){if(e.blockedOn!==null)return!1;for(var n=e.targetContainers;0<n.length;){var r=eo(e.domEventName,e.eventSystemFlags,n[0],e.nativeEvent);if(r===null){r=e.nativeEvent;var s=new r.constructor(r.type,r);Oe=s,r.target.dispatchEvent(s),Oe=null}else return n=us(r),n!==null&&Ka(n),e.blockedOn=r,!1;n.shift()}return!0}function gi(e,n,r){As(e)&&r.delete(n)}function Bu(){Ja=!1,En!==null&&As(En)&&(En=null),In!==null&&As(In)&&(In=null),Pn!==null&&As(Pn)&&(Pn=null),Yr.forEach(gi),Xr.forEach(gi)}function Jr(e,n){e.blockedOn===n&&(e.blockedOn=null,Ja||(Ja=!0,x.unstable_scheduleCallback(x.unstable_NormalPriority,Bu)))}function Zr(e){function n(a){return Jr(a,e)}if(0<Os.length){Jr(Os[0],e);for(var r=1;r<Os.length;r++){var s=Os[r];s.blockedOn===e&&(s.blockedOn=null)}}for(En!==null&&Jr(En,e),In!==null&&Jr(In,e),Pn!==null&&Jr(Pn,e),Yr.forEach(n),Xr.forEach(n),r=0;r<Tn.length;r++)s=Tn[r],s.blockedOn===e&&(s.blockedOn=null);for(;0<Tn.length&&(r=Tn[0],r.blockedOn===null);)hi(r),r.blockedOn===null&&Tn.shift()}var vr=j.ReactCurrentBatchConfig,$s=!0;function Wu(e,n,r,s){var a=Be,o=vr.transition;vr.transition=null;try{Be=1,Za(e,n,r,s)}finally{Be=a,vr.transition=o}}function Gu(e,n,r,s){var a=Be,o=vr.transition;vr.transition=null;try{Be=4,Za(e,n,r,s)}finally{Be=a,vr.transition=o}}function Za(e,n,r,s){if($s){var a=eo(e,n,r,s);if(a===null)vo(e,n,s,Us,r),xi(e,s);else if(Vu(a,e,n,r,s))s.stopPropagation();else if(xi(e,s),n&4&&-1<Uu.indexOf(e)){for(;a!==null;){var o=us(a);if(o!==null&&ui(o),o=eo(e,n,r,s),o===null&&vo(e,n,s,Us,r),o===a)break;a=o}a!==null&&s.stopPropagation()}else vo(e,n,s,null,r)}}var Us=null;function eo(e,n,r,s){if(Us=null,e=ct(s),e=tr(e),e!==null)if(n=er(e),n===null)e=null;else if(r=n.tag,r===13){if(e=ti(n),e!==null)return e;e=null}else if(r===3){if(n.stateNode.current.memoizedState.isDehydrated)return n.tag===3?n.stateNode.containerInfo:null;e=null}else n!==e&&(e=null);return Us=e,null}function vi(e){switch(e){case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 1;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"toggle":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 4;case"message":switch(Tu()){case qa:return 1;case li:return 4;case Rs:case Ru:return 16;case ii:return 536870912;default:return 16}default:return 16}}var Rn=null,to=null,Vs=null;function yi(){if(Vs)return Vs;var e,n=to,r=n.length,s,a="value"in Rn?Rn.value:Rn.textContent,o=a.length;for(e=0;e<r&&n[e]===a[e];e++);var i=r-e;for(s=1;s<=i&&n[r-s]===a[o-s];s++);return Vs=a.slice(e,1<s?1-s:void 0)}function Bs(e){var n=e.keyCode;return"charCode"in e?(e=e.charCode,e===0&&n===13&&(e=13)):e=n,e===10&&(e=13),32<=e||e===13?e:0}function Ws(){return!0}function bi(){return!1}function Ft(e){function n(r,s,a,o,i){this._reactName=r,this._targetInst=a,this.type=s,this.nativeEvent=o,this.target=i,this.currentTarget=null;for(var f in e)e.hasOwnProperty(f)&&(r=e[f],this[f]=r?r(o):o[f]);return this.isDefaultPrevented=(o.defaultPrevented!=null?o.defaultPrevented:o.returnValue===!1)?Ws:bi,this.isPropagationStopped=bi,this}return J(n.prototype,{preventDefault:function(){this.defaultPrevented=!0;var r=this.nativeEvent;r&&(r.preventDefault?r.preventDefault():typeof r.returnValue!="unknown"&&(r.returnValue=!1),this.isDefaultPrevented=Ws)},stopPropagation:function(){var r=this.nativeEvent;r&&(r.stopPropagation?r.stopPropagation():typeof r.cancelBubble!="unknown"&&(r.cancelBubble=!0),this.isPropagationStopped=Ws)},persist:function(){},isPersistent:Ws}),n}var yr={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},no=Ft(yr),es=J({},yr,{view:0,detail:0}),Hu=Ft(es),ro,so,ts,Gs=J({},es,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:oo,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return"movementX"in e?e.movementX:(e!==ts&&(ts&&e.type==="mousemove"?(ro=e.screenX-ts.screenX,so=e.screenY-ts.screenY):so=ro=0,ts=e),ro)},movementY:function(e){return"movementY"in e?e.movementY:so}}),ji=Ft(Gs),qu=J({},Gs,{dataTransfer:0}),Qu=Ft(qu),Yu=J({},es,{relatedTarget:0}),ao=Ft(Yu),Xu=J({},yr,{animationName:0,elapsedTime:0,pseudoElement:0}),Ku=Ft(Xu),Ju=J({},yr,{clipboardData:function(e){return"clipboardData"in e?e.clipboardData:window.clipboardData}}),Zu=Ft(Ju),ep=J({},yr,{data:0}),wi=Ft(ep),tp={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},np={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},rp={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function sp(e){var n=this.nativeEvent;return n.getModifierState?n.getModifierState(e):(e=rp[e])?!!n[e]:!1}function oo(){return sp}var ap=J({},es,{key:function(e){if(e.key){var n=tp[e.key]||e.key;if(n!=="Unidentified")return n}return e.type==="keypress"?(e=Bs(e),e===13?"Enter":String.fromCharCode(e)):e.type==="keydown"||e.type==="keyup"?np[e.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:oo,charCode:function(e){return e.type==="keypress"?Bs(e):0},keyCode:function(e){return e.type==="keydown"||e.type==="keyup"?e.keyCode:0},which:function(e){return e.type==="keypress"?Bs(e):e.type==="keydown"||e.type==="keyup"?e.keyCode:0}}),op=Ft(ap),lp=J({},Gs,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),ki=Ft(lp),ip=J({},es,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:oo}),cp=Ft(ip),dp=J({},yr,{propertyName:0,elapsedTime:0,pseudoElement:0}),up=Ft(dp),pp=J({},Gs,{deltaX:function(e){return"deltaX"in e?e.deltaX:"wheelDeltaX"in e?-e.wheelDeltaX:0},deltaY:function(e){return"deltaY"in e?e.deltaY:"wheelDeltaY"in e?-e.wheelDeltaY:"wheelDelta"in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0}),fp=Ft(pp),mp=[9,13,27,32],lo=R&&"CompositionEvent"in window,ns=null;R&&"documentMode"in document&&(ns=document.documentMode);var xp=R&&"TextEvent"in window&&!ns,Si=R&&(!lo||ns&&8<ns&&11>=ns),Ni=" ",Ci=!1;function _i(e,n){switch(e){case"keyup":return mp.indexOf(n.keyCode)!==-1;case"keydown":return n.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function zi(e){return e=e.detail,typeof e=="object"&&"data"in e?e.data:null}var br=!1;function hp(e,n){switch(e){case"compositionend":return zi(n);case"keypress":return n.which!==32?null:(Ci=!0,Ni);case"textInput":return e=n.data,e===Ni&&Ci?null:e;default:return null}}function gp(e,n){if(br)return e==="compositionend"||!lo&&_i(e,n)?(e=yi(),Vs=to=Rn=null,br=!1,e):null;switch(e){case"paste":return null;case"keypress":if(!(n.ctrlKey||n.altKey||n.metaKey)||n.ctrlKey&&n.altKey){if(n.char&&1<n.char.length)return n.char;if(n.which)return String.fromCharCode(n.which)}return null;case"compositionend":return Si&&n.locale!=="ko"?null:n.data;default:return null}}var vp={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function Ei(e){var n=e&&e.nodeName&&e.nodeName.toLowerCase();return n==="input"?!!vp[e.type]:n==="textarea"}function Ii(e,n,r,s){Gr(s),n=Xs(n,"onChange"),0<n.length&&(r=new no("onChange","change",null,r,s),e.push({event:r,listeners:n}))}var rs=null,ss=null;function yp(e){Qi(e,0)}function Hs(e){var n=Nr(e);if(pe(n))return e}function bp(e,n){if(e==="change")return n}var Pi=!1;if(R){var io;if(R){var co="oninput"in document;if(!co){var Ti=document.createElement("div");Ti.setAttribute("oninput","return;"),co=typeof Ti.oninput=="function"}io=co}else io=!1;Pi=io&&(!document.documentMode||9<document.documentMode)}function Ri(){rs&&(rs.detachEvent("onpropertychange",Mi),ss=rs=null)}function Mi(e){if(e.propertyName==="value"&&Hs(ss)){var n=[];Ii(n,ss,e,ct(e)),be(yp,n)}}function jp(e,n,r){e==="focusin"?(Ri(),rs=n,ss=r,rs.attachEvent("onpropertychange",Mi)):e==="focusout"&&Ri()}function wp(e){if(e==="selectionchange"||e==="keyup"||e==="keydown")return Hs(ss)}function kp(e,n){if(e==="click")return Hs(n)}function Sp(e,n){if(e==="input"||e==="change")return Hs(n)}function Np(e,n){return e===n&&(e!==0||1/e===1/n)||e!==e&&n!==n}var Kt=typeof Object.is=="function"?Object.is:Np;function as(e,n){if(Kt(e,n))return!0;if(typeof e!="object"||e===null||typeof n!="object"||n===null)return!1;var r=Object.keys(e),s=Object.keys(n);if(r.length!==s.length)return!1;for(s=0;s<r.length;s++){var a=r[s];if(!I.call(n,a)||!Kt(e[a],n[a]))return!1}return!0}function Li(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function Fi(e,n){var r=Li(e);e=0;for(var s;r;){if(r.nodeType===3){if(s=e+r.textContent.length,e<=n&&s>=n)return{node:r,offset:n-e};e=s}e:{for(;r;){if(r.nextSibling){r=r.nextSibling;break e}r=r.parentNode}r=void 0}r=Li(r)}}function Di(e,n){return e&&n?e===n?!0:e&&e.nodeType===3?!1:n&&n.nodeType===3?Di(e,n.parentNode):"contains"in e?e.contains(n):e.compareDocumentPosition?!!(e.compareDocumentPosition(n)&16):!1:!1}function Oi(){for(var e=window,n=Ne();n instanceof e.HTMLIFrameElement;){try{var r=typeof n.contentWindow.location.href=="string"}catch{r=!1}if(r)e=n.contentWindow;else break;n=Ne(e.document)}return n}function uo(e){var n=e&&e.nodeName&&e.nodeName.toLowerCase();return n&&(n==="input"&&(e.type==="text"||e.type==="search"||e.type==="tel"||e.type==="url"||e.type==="password")||n==="textarea"||e.contentEditable==="true")}function Cp(e){var n=Oi(),r=e.focusedElem,s=e.selectionRange;if(n!==r&&r&&r.ownerDocument&&Di(r.ownerDocument.documentElement,r)){if(s!==null&&uo(r)){if(n=s.start,e=s.end,e===void 0&&(e=n),"selectionStart"in r)r.selectionStart=n,r.selectionEnd=Math.min(e,r.value.length);else if(e=(n=r.ownerDocument||document)&&n.defaultView||window,e.getSelection){e=e.getSelection();var a=r.textContent.length,o=Math.min(s.start,a);s=s.end===void 0?o:Math.min(s.end,a),!e.extend&&o>s&&(a=s,s=o,o=a),a=Fi(r,o);var i=Fi(r,s);a&&i&&(e.rangeCount!==1||e.anchorNode!==a.node||e.anchorOffset!==a.offset||e.focusNode!==i.node||e.focusOffset!==i.offset)&&(n=n.createRange(),n.setStart(a.node,a.offset),e.removeAllRanges(),o>s?(e.addRange(n),e.extend(i.node,i.offset)):(n.setEnd(i.node,i.offset),e.addRange(n)))}}for(n=[],e=r;e=e.parentNode;)e.nodeType===1&&n.push({element:e,left:e.scrollLeft,top:e.scrollTop});for(typeof r.focus=="function"&&r.focus(),r=0;r<n.length;r++)e=n[r],e.element.scrollLeft=e.left,e.element.scrollTop=e.top}}var _p=R&&"documentMode"in document&&11>=document.documentMode,jr=null,po=null,os=null,fo=!1;function Ai(e,n,r){var s=r.window===r?r.document:r.nodeType===9?r:r.ownerDocument;fo||jr==null||jr!==Ne(s)||(s=jr,"selectionStart"in s&&uo(s)?s={start:s.selectionStart,end:s.selectionEnd}:(s=(s.ownerDocument&&s.ownerDocument.defaultView||window).getSelection(),s={anchorNode:s.anchorNode,anchorOffset:s.anchorOffset,focusNode:s.focusNode,focusOffset:s.focusOffset}),os&&as(os,s)||(os=s,s=Xs(po,"onSelect"),0<s.length&&(n=new no("onSelect","select",null,n,r),e.push({event:n,listeners:s}),n.target=jr)))}function qs(e,n){var r={};return r[e.toLowerCase()]=n.toLowerCase(),r["Webkit"+e]="webkit"+n,r["Moz"+e]="moz"+n,r}var wr={animationend:qs("Animation","AnimationEnd"),animationiteration:qs("Animation","AnimationIteration"),animationstart:qs("Animation","AnimationStart"),transitionend:qs("Transition","TransitionEnd")},mo={},$i={};R&&($i=document.createElement("div").style,"AnimationEvent"in window||(delete wr.animationend.animation,delete wr.animationiteration.animation,delete wr.animationstart.animation),"TransitionEvent"in window||delete wr.transitionend.transition);function Qs(e){if(mo[e])return mo[e];if(!wr[e])return e;var n=wr[e],r;for(r in n)if(n.hasOwnProperty(r)&&r in $i)return mo[e]=n[r];return e}var Ui=Qs("animationend"),Vi=Qs("animationiteration"),Bi=Qs("animationstart"),Wi=Qs("transitionend"),Gi=new Map,Hi="abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");function Mn(e,n){Gi.set(e,n),b(n,[e])}for(var xo=0;xo<Hi.length;xo++){var ho=Hi[xo],zp=ho.toLowerCase(),Ep=ho[0].toUpperCase()+ho.slice(1);Mn(zp,"on"+Ep)}Mn(Ui,"onAnimationEnd"),Mn(Vi,"onAnimationIteration"),Mn(Bi,"onAnimationStart"),Mn("dblclick","onDoubleClick"),Mn("focusin","onFocus"),Mn("focusout","onBlur"),Mn(Wi,"onTransitionEnd"),S("onMouseEnter",["mouseout","mouseover"]),S("onMouseLeave",["mouseout","mouseover"]),S("onPointerEnter",["pointerout","pointerover"]),S("onPointerLeave",["pointerout","pointerover"]),b("onChange","change click focusin focusout input keydown keyup selectionchange".split(" ")),b("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" ")),b("onBeforeInput",["compositionend","keypress","textInput","paste"]),b("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" ")),b("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" ")),b("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var ls="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),Ip=new Set("cancel close invalid load scroll toggle".split(" ").concat(ls));function qi(e,n,r){var s=e.type||"unknown-event";e.currentTarget=r,zu(s,n,void 0,e),e.currentTarget=null}function Qi(e,n){n=(n&4)!==0;for(var r=0;r<e.length;r++){var s=e[r],a=s.event;s=s.listeners;e:{var o=void 0;if(n)for(var i=s.length-1;0<=i;i--){var f=s[i],g=f.instance,O=f.currentTarget;if(f=f.listener,g!==o&&a.isPropagationStopped())break e;qi(a,f,O),o=g}else for(i=0;i<s.length;i++){if(f=s[i],g=f.instance,O=f.currentTarget,f=f.listener,g!==o&&a.isPropagationStopped())break e;qi(a,f,O),o=g}}}if(Ts)throw e=Ha,Ts=!1,Ha=null,e}function Ye(e,n){var r=n[So];r===void 0&&(r=n[So]=new Set);var s=e+"__bubble";r.has(s)||(Yi(n,e,2,!1),r.add(s))}function go(e,n,r){var s=0;n&&(s|=4),Yi(r,e,s,n)}var Ys="_reactListening"+Math.random().toString(36).slice(2);function is(e){if(!e[Ys]){e[Ys]=!0,N.forEach(function(r){r!=="selectionchange"&&(Ip.has(r)||go(r,!1,e),go(r,!0,e))});var n=e.nodeType===9?e:e.ownerDocument;n===null||n[Ys]||(n[Ys]=!0,go("selectionchange",!1,n))}}function Yi(e,n,r,s){switch(vi(n)){case 1:var a=Wu;break;case 4:a=Gu;break;default:a=Za}r=a.bind(null,n,r,e),a=void 0,!Le||n!=="touchstart"&&n!=="touchmove"&&n!=="wheel"||(a=!0),s?a!==void 0?e.addEventListener(n,r,{capture:!0,passive:a}):e.addEventListener(n,r,!0):a!==void 0?e.addEventListener(n,r,{passive:a}):e.addEventListener(n,r,!1)}function vo(e,n,r,s,a){var o=s;if((n&1)===0&&(n&2)===0&&s!==null)e:for(;;){if(s===null)return;var i=s.tag;if(i===3||i===4){var f=s.stateNode.containerInfo;if(f===a||f.nodeType===8&&f.parentNode===a)break;if(i===4)for(i=s.return;i!==null;){var g=i.tag;if((g===3||g===4)&&(g=i.stateNode.containerInfo,g===a||g.nodeType===8&&g.parentNode===a))return;i=i.return}for(;f!==null;){if(i=tr(f),i===null)return;if(g=i.tag,g===5||g===6){s=o=i;continue e}f=f.parentNode}}s=s.return}be(function(){var O=o,se=ct(r),oe=[];e:{var ne=Gi.get(e);if(ne!==void 0){var je=no,Se=e;switch(e){case"keypress":if(Bs(r)===0)break e;case"keydown":case"keyup":je=op;break;case"focusin":Se="focus",je=ao;break;case"focusout":Se="blur",je=ao;break;case"beforeblur":case"afterblur":je=ao;break;case"click":if(r.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":je=ji;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":je=Qu;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":je=cp;break;case Ui:case Vi:case Bi:je=Ku;break;case Wi:je=up;break;case"scroll":je=Hu;break;case"wheel":je=fp;break;case"copy":case"cut":case"paste":je=Zu;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":je=ki}var Ce=(n&4)!==0,st=!Ce&&e==="scroll",z=Ce?ne!==null?ne+"Capture":null:ne;Ce=[];for(var y=O,M;y!==null;){M=y;var de=M.stateNode;if(M.tag===5&&de!==null&&(M=de,z!==null&&(de=Te(y,z),de!=null&&Ce.push(cs(y,de,M)))),st)break;y=y.return}0<Ce.length&&(ne=new je(ne,Se,null,r,se),oe.push({event:ne,listeners:Ce}))}}if((n&7)===0){e:{if(ne=e==="mouseover"||e==="pointerover",je=e==="mouseout"||e==="pointerout",ne&&r!==Oe&&(Se=r.relatedTarget||r.fromElement)&&(tr(Se)||Se[xn]))break e;if((je||ne)&&(ne=se.window===se?se:(ne=se.ownerDocument)?ne.defaultView||ne.parentWindow:window,je?(Se=r.relatedTarget||r.toElement,je=O,Se=Se?tr(Se):null,Se!==null&&(st=er(Se),Se!==st||Se.tag!==5&&Se.tag!==6)&&(Se=null)):(je=null,Se=O),je!==Se)){if(Ce=ji,de="onMouseLeave",z="onMouseEnter",y="mouse",(e==="pointerout"||e==="pointerover")&&(Ce=ki,de="onPointerLeave",z="onPointerEnter",y="pointer"),st=je==null?ne:Nr(je),M=Se==null?ne:Nr(Se),ne=new Ce(de,y+"leave",je,r,se),ne.target=st,ne.relatedTarget=M,de=null,tr(se)===O&&(Ce=new Ce(z,y+"enter",Se,r,se),Ce.target=M,Ce.relatedTarget=st,de=Ce),st=de,je&&Se)t:{for(Ce=je,z=Se,y=0,M=Ce;M;M=kr(M))y++;for(M=0,de=z;de;de=kr(de))M++;for(;0<y-M;)Ce=kr(Ce),y--;for(;0<M-y;)z=kr(z),M--;for(;y--;){if(Ce===z||z!==null&&Ce===z.alternate)break t;Ce=kr(Ce),z=kr(z)}Ce=null}else Ce=null;je!==null&&Xi(oe,ne,je,Ce,!1),Se!==null&&st!==null&&Xi(oe,st,Se,Ce,!0)}}e:{if(ne=O?Nr(O):window,je=ne.nodeName&&ne.nodeName.toLowerCase(),je==="select"||je==="input"&&ne.type==="file")var _e=bp;else if(Ei(ne))if(Pi)_e=Sp;else{_e=wp;var Ee=jp}else(je=ne.nodeName)&&je.toLowerCase()==="input"&&(ne.type==="checkbox"||ne.type==="radio")&&(_e=kp);if(_e&&(_e=_e(e,O))){Ii(oe,_e,r,se);break e}Ee&&Ee(e,ne,O),e==="focusout"&&(Ee=ne._wrapperState)&&Ee.controlled&&ne.type==="number"&&_t(ne,"number",ne.value)}switch(Ee=O?Nr(O):window,e){case"focusin":(Ei(Ee)||Ee.contentEditable==="true")&&(jr=Ee,po=O,os=null);break;case"focusout":os=po=jr=null;break;case"mousedown":fo=!0;break;case"contextmenu":case"mouseup":case"dragend":fo=!1,Ai(oe,r,se);break;case"selectionchange":if(_p)break;case"keydown":case"keyup":Ai(oe,r,se)}var Ie;if(lo)e:{switch(e){case"compositionstart":var Pe="onCompositionStart";break e;case"compositionend":Pe="onCompositionEnd";break e;case"compositionupdate":Pe="onCompositionUpdate";break e}Pe=void 0}else br?_i(e,r)&&(Pe="onCompositionEnd"):e==="keydown"&&r.keyCode===229&&(Pe="onCompositionStart");Pe&&(Si&&r.locale!=="ko"&&(br||Pe!=="onCompositionStart"?Pe==="onCompositionEnd"&&br&&(Ie=yi()):(Rn=se,to="value"in Rn?Rn.value:Rn.textContent,br=!0)),Ee=Xs(O,Pe),0<Ee.length&&(Pe=new wi(Pe,e,null,r,se),oe.push({event:Pe,listeners:Ee}),Ie?Pe.data=Ie:(Ie=zi(r),Ie!==null&&(Pe.data=Ie)))),(Ie=xp?hp(e,r):gp(e,r))&&(O=Xs(O,"onBeforeInput"),0<O.length&&(se=new wi("onBeforeInput","beforeinput",null,r,se),oe.push({event:se,listeners:O}),se.data=Ie))}Qi(oe,n)})}function cs(e,n,r){return{instance:e,listener:n,currentTarget:r}}function Xs(e,n){for(var r=n+"Capture",s=[];e!==null;){var a=e,o=a.stateNode;a.tag===5&&o!==null&&(a=o,o=Te(e,r),o!=null&&s.unshift(cs(e,o,a)),o=Te(e,n),o!=null&&s.push(cs(e,o,a))),e=e.return}return s}function kr(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5);return e||null}function Xi(e,n,r,s,a){for(var o=n._reactName,i=[];r!==null&&r!==s;){var f=r,g=f.alternate,O=f.stateNode;if(g!==null&&g===s)break;f.tag===5&&O!==null&&(f=O,a?(g=Te(r,o),g!=null&&i.unshift(cs(r,g,f))):a||(g=Te(r,o),g!=null&&i.push(cs(r,g,f)))),r=r.return}i.length!==0&&e.push({event:n,listeners:i})}var Pp=/\r\n?/g,Tp=/\u0000|\uFFFD/g;function Ki(e){return(typeof e=="string"?e:""+e).replace(Pp,`
`).replace(Tp,"")}function Ks(e,n,r){if(n=Ki(n),Ki(e)!==n&&r)throw Error(d(425))}function Js(){}var yo=null,bo=null;function jo(e,n){return e==="textarea"||e==="noscript"||typeof n.children=="string"||typeof n.children=="number"||typeof n.dangerouslySetInnerHTML=="object"&&n.dangerouslySetInnerHTML!==null&&n.dangerouslySetInnerHTML.__html!=null}var wo=typeof setTimeout=="function"?setTimeout:void 0,Rp=typeof clearTimeout=="function"?clearTimeout:void 0,Ji=typeof Promise=="function"?Promise:void 0,Mp=typeof queueMicrotask=="function"?queueMicrotask:typeof Ji<"u"?function(e){return Ji.resolve(null).then(e).catch(Lp)}:wo;function Lp(e){setTimeout(function(){throw e})}function ko(e,n){var r=n,s=0;do{var a=r.nextSibling;if(e.removeChild(r),a&&a.nodeType===8)if(r=a.data,r==="/$"){if(s===0){e.removeChild(a),Zr(n);return}s--}else r!=="$"&&r!=="$?"&&r!=="$!"||s++;r=a}while(r);Zr(n)}function Ln(e){for(;e!=null;e=e.nextSibling){var n=e.nodeType;if(n===1||n===3)break;if(n===8){if(n=e.data,n==="$"||n==="$!"||n==="$?")break;if(n==="/$")return null}}return e}function Zi(e){e=e.previousSibling;for(var n=0;e;){if(e.nodeType===8){var r=e.data;if(r==="$"||r==="$!"||r==="$?"){if(n===0)return e;n--}else r==="/$"&&n++}e=e.previousSibling}return null}var Sr=Math.random().toString(36).slice(2),an="__reactFiber$"+Sr,ds="__reactProps$"+Sr,xn="__reactContainer$"+Sr,So="__reactEvents$"+Sr,Fp="__reactListeners$"+Sr,Dp="__reactHandles$"+Sr;function tr(e){var n=e[an];if(n)return n;for(var r=e.parentNode;r;){if(n=r[xn]||r[an]){if(r=n.alternate,n.child!==null||r!==null&&r.child!==null)for(e=Zi(e);e!==null;){if(r=e[an])return r;e=Zi(e)}return n}e=r,r=e.parentNode}return null}function us(e){return e=e[an]||e[xn],!e||e.tag!==5&&e.tag!==6&&e.tag!==13&&e.tag!==3?null:e}function Nr(e){if(e.tag===5||e.tag===6)return e.stateNode;throw Error(d(33))}function Zs(e){return e[ds]||null}var No=[],Cr=-1;function Fn(e){return{current:e}}function Xe(e){0>Cr||(e.current=No[Cr],No[Cr]=null,Cr--)}function qe(e,n){Cr++,No[Cr]=e.current,e.current=n}var Dn={},yt=Fn(Dn),zt=Fn(!1),nr=Dn;function _r(e,n){var r=e.type.contextTypes;if(!r)return Dn;var s=e.stateNode;if(s&&s.__reactInternalMemoizedUnmaskedChildContext===n)return s.__reactInternalMemoizedMaskedChildContext;var a={},o;for(o in r)a[o]=n[o];return s&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=n,e.__reactInternalMemoizedMaskedChildContext=a),a}function Et(e){return e=e.childContextTypes,e!=null}function ea(){Xe(zt),Xe(yt)}function ec(e,n,r){if(yt.current!==Dn)throw Error(d(168));qe(yt,n),qe(zt,r)}function tc(e,n,r){var s=e.stateNode;if(n=n.childContextTypes,typeof s.getChildContext!="function")return r;s=s.getChildContext();for(var a in s)if(!(a in n))throw Error(d(108,Y(e)||"Unknown",a));return J({},r,s)}function ta(e){return e=(e=e.stateNode)&&e.__reactInternalMemoizedMergedChildContext||Dn,nr=yt.current,qe(yt,e),qe(zt,zt.current),!0}function nc(e,n,r){var s=e.stateNode;if(!s)throw Error(d(169));r?(e=tc(e,n,nr),s.__reactInternalMemoizedMergedChildContext=e,Xe(zt),Xe(yt),qe(yt,e)):Xe(zt),qe(zt,r)}var hn=null,na=!1,Co=!1;function rc(e){hn===null?hn=[e]:hn.push(e)}function Op(e){na=!0,rc(e)}function On(){if(!Co&&hn!==null){Co=!0;var e=0,n=Be;try{var r=hn;for(Be=1;e<r.length;e++){var s=r[e];do s=s(!0);while(s!==null)}hn=null,na=!1}catch(a){throw hn!==null&&(hn=hn.slice(e+1)),ai(qa,On),a}finally{Be=n,Co=!1}}return null}var zr=[],Er=0,ra=null,sa=0,Ut=[],Vt=0,rr=null,gn=1,vn="";function sr(e,n){zr[Er++]=sa,zr[Er++]=ra,ra=e,sa=n}function sc(e,n,r){Ut[Vt++]=gn,Ut[Vt++]=vn,Ut[Vt++]=rr,rr=e;var s=gn;e=vn;var a=32-Xt(s)-1;s&=~(1<<a),r+=1;var o=32-Xt(n)+a;if(30<o){var i=a-a%5;o=(s&(1<<i)-1).toString(32),s>>=i,a-=i,gn=1<<32-Xt(n)+a|r<<a|s,vn=o+e}else gn=1<<o|r<<a|s,vn=e}function _o(e){e.return!==null&&(sr(e,1),sc(e,1,0))}function zo(e){for(;e===ra;)ra=zr[--Er],zr[Er]=null,sa=zr[--Er],zr[Er]=null;for(;e===rr;)rr=Ut[--Vt],Ut[Vt]=null,vn=Ut[--Vt],Ut[Vt]=null,gn=Ut[--Vt],Ut[Vt]=null}var Dt=null,Ot=null,Ze=!1,Jt=null;function ac(e,n){var r=Ht(5,null,null,0);r.elementType="DELETED",r.stateNode=n,r.return=e,n=e.deletions,n===null?(e.deletions=[r],e.flags|=16):n.push(r)}function oc(e,n){switch(e.tag){case 5:var r=e.type;return n=n.nodeType!==1||r.toLowerCase()!==n.nodeName.toLowerCase()?null:n,n!==null?(e.stateNode=n,Dt=e,Ot=Ln(n.firstChild),!0):!1;case 6:return n=e.pendingProps===""||n.nodeType!==3?null:n,n!==null?(e.stateNode=n,Dt=e,Ot=null,!0):!1;case 13:return n=n.nodeType!==8?null:n,n!==null?(r=rr!==null?{id:gn,overflow:vn}:null,e.memoizedState={dehydrated:n,treeContext:r,retryLane:1073741824},r=Ht(18,null,null,0),r.stateNode=n,r.return=e,e.child=r,Dt=e,Ot=null,!0):!1;default:return!1}}function Eo(e){return(e.mode&1)!==0&&(e.flags&128)===0}function Io(e){if(Ze){var n=Ot;if(n){var r=n;if(!oc(e,n)){if(Eo(e))throw Error(d(418));n=Ln(r.nextSibling);var s=Dt;n&&oc(e,n)?ac(s,r):(e.flags=e.flags&-4097|2,Ze=!1,Dt=e)}}else{if(Eo(e))throw Error(d(418));e.flags=e.flags&-4097|2,Ze=!1,Dt=e}}}function lc(e){for(e=e.return;e!==null&&e.tag!==5&&e.tag!==3&&e.tag!==13;)e=e.return;Dt=e}function aa(e){if(e!==Dt)return!1;if(!Ze)return lc(e),Ze=!0,!1;var n;if((n=e.tag!==3)&&!(n=e.tag!==5)&&(n=e.type,n=n!=="head"&&n!=="body"&&!jo(e.type,e.memoizedProps)),n&&(n=Ot)){if(Eo(e))throw ic(),Error(d(418));for(;n;)ac(e,n),n=Ln(n.nextSibling)}if(lc(e),e.tag===13){if(e=e.memoizedState,e=e!==null?e.dehydrated:null,!e)throw Error(d(317));e:{for(e=e.nextSibling,n=0;e;){if(e.nodeType===8){var r=e.data;if(r==="/$"){if(n===0){Ot=Ln(e.nextSibling);break e}n--}else r!=="$"&&r!=="$!"&&r!=="$?"||n++}e=e.nextSibling}Ot=null}}else Ot=Dt?Ln(e.stateNode.nextSibling):null;return!0}function ic(){for(var e=Ot;e;)e=Ln(e.nextSibling)}function Ir(){Ot=Dt=null,Ze=!1}function Po(e){Jt===null?Jt=[e]:Jt.push(e)}var Ap=j.ReactCurrentBatchConfig;function ps(e,n,r){if(e=r.ref,e!==null&&typeof e!="function"&&typeof e!="object"){if(r._owner){if(r=r._owner,r){if(r.tag!==1)throw Error(d(309));var s=r.stateNode}if(!s)throw Error(d(147,e));var a=s,o=""+e;return n!==null&&n.ref!==null&&typeof n.ref=="function"&&n.ref._stringRef===o?n.ref:(n=function(i){var f=a.refs;i===null?delete f[o]:f[o]=i},n._stringRef=o,n)}if(typeof e!="string")throw Error(d(284));if(!r._owner)throw Error(d(290,e))}return e}function oa(e,n){throw e=Object.prototype.toString.call(n),Error(d(31,e==="[object Object]"?"object with keys {"+Object.keys(n).join(", ")+"}":e))}function cc(e){var n=e._init;return n(e._payload)}function dc(e){function n(z,y){if(e){var M=z.deletions;M===null?(z.deletions=[y],z.flags|=16):M.push(y)}}function r(z,y){if(!e)return null;for(;y!==null;)n(z,y),y=y.sibling;return null}function s(z,y){for(z=new Map;y!==null;)y.key!==null?z.set(y.key,y):z.set(y.index,y),y=y.sibling;return z}function a(z,y){return z=Hn(z,y),z.index=0,z.sibling=null,z}function o(z,y,M){return z.index=M,e?(M=z.alternate,M!==null?(M=M.index,M<y?(z.flags|=2,y):M):(z.flags|=2,y)):(z.flags|=1048576,y)}function i(z){return e&&z.alternate===null&&(z.flags|=2),z}function f(z,y,M,de){return y===null||y.tag!==6?(y=wl(M,z.mode,de),y.return=z,y):(y=a(y,M),y.return=z,y)}function g(z,y,M,de){var _e=M.type;return _e===h?se(z,y,M.props.children,de,M.key):y!==null&&(y.elementType===_e||typeof _e=="object"&&_e!==null&&_e.$$typeof===ie&&cc(_e)===y.type)?(de=a(y,M.props),de.ref=ps(z,y,M),de.return=z,de):(de=Ia(M.type,M.key,M.props,null,z.mode,de),de.ref=ps(z,y,M),de.return=z,de)}function O(z,y,M,de){return y===null||y.tag!==4||y.stateNode.containerInfo!==M.containerInfo||y.stateNode.implementation!==M.implementation?(y=kl(M,z.mode,de),y.return=z,y):(y=a(y,M.children||[]),y.return=z,y)}function se(z,y,M,de,_e){return y===null||y.tag!==7?(y=pr(M,z.mode,de,_e),y.return=z,y):(y=a(y,M),y.return=z,y)}function oe(z,y,M){if(typeof y=="string"&&y!==""||typeof y=="number")return y=wl(""+y,z.mode,M),y.return=z,y;if(typeof y=="object"&&y!==null){switch(y.$$typeof){case k:return M=Ia(y.type,y.key,y.props,null,z.mode,M),M.ref=ps(z,null,y),M.return=z,M;case B:return y=kl(y,z.mode,M),y.return=z,y;case ie:var de=y._init;return oe(z,de(y._payload),M)}if(We(y)||X(y))return y=pr(y,z.mode,M,null),y.return=z,y;oa(z,y)}return null}function ne(z,y,M,de){var _e=y!==null?y.key:null;if(typeof M=="string"&&M!==""||typeof M=="number")return _e!==null?null:f(z,y,""+M,de);if(typeof M=="object"&&M!==null){switch(M.$$typeof){case k:return M.key===_e?g(z,y,M,de):null;case B:return M.key===_e?O(z,y,M,de):null;case ie:return _e=M._init,ne(z,y,_e(M._payload),de)}if(We(M)||X(M))return _e!==null?null:se(z,y,M,de,null);oa(z,M)}return null}function je(z,y,M,de,_e){if(typeof de=="string"&&de!==""||typeof de=="number")return z=z.get(M)||null,f(y,z,""+de,_e);if(typeof de=="object"&&de!==null){switch(de.$$typeof){case k:return z=z.get(de.key===null?M:de.key)||null,g(y,z,de,_e);case B:return z=z.get(de.key===null?M:de.key)||null,O(y,z,de,_e);case ie:var Ee=de._init;return je(z,y,M,Ee(de._payload),_e)}if(We(de)||X(de))return z=z.get(M)||null,se(y,z,de,_e,null);oa(y,de)}return null}function Se(z,y,M,de){for(var _e=null,Ee=null,Ie=y,Pe=y=0,xt=null;Ie!==null&&Pe<M.length;Pe++){Ie.index>Pe?(xt=Ie,Ie=null):xt=Ie.sibling;var Ue=ne(z,Ie,M[Pe],de);if(Ue===null){Ie===null&&(Ie=xt);break}e&&Ie&&Ue.alternate===null&&n(z,Ie),y=o(Ue,y,Pe),Ee===null?_e=Ue:Ee.sibling=Ue,Ee=Ue,Ie=xt}if(Pe===M.length)return r(z,Ie),Ze&&sr(z,Pe),_e;if(Ie===null){for(;Pe<M.length;Pe++)Ie=oe(z,M[Pe],de),Ie!==null&&(y=o(Ie,y,Pe),Ee===null?_e=Ie:Ee.sibling=Ie,Ee=Ie);return Ze&&sr(z,Pe),_e}for(Ie=s(z,Ie);Pe<M.length;Pe++)xt=je(Ie,z,Pe,M[Pe],de),xt!==null&&(e&&xt.alternate!==null&&Ie.delete(xt.key===null?Pe:xt.key),y=o(xt,y,Pe),Ee===null?_e=xt:Ee.sibling=xt,Ee=xt);return e&&Ie.forEach(function(qn){return n(z,qn)}),Ze&&sr(z,Pe),_e}function Ce(z,y,M,de){var _e=X(M);if(typeof _e!="function")throw Error(d(150));if(M=_e.call(M),M==null)throw Error(d(151));for(var Ee=_e=null,Ie=y,Pe=y=0,xt=null,Ue=M.next();Ie!==null&&!Ue.done;Pe++,Ue=M.next()){Ie.index>Pe?(xt=Ie,Ie=null):xt=Ie.sibling;var qn=ne(z,Ie,Ue.value,de);if(qn===null){Ie===null&&(Ie=xt);break}e&&Ie&&qn.alternate===null&&n(z,Ie),y=o(qn,y,Pe),Ee===null?_e=qn:Ee.sibling=qn,Ee=qn,Ie=xt}if(Ue.done)return r(z,Ie),Ze&&sr(z,Pe),_e;if(Ie===null){for(;!Ue.done;Pe++,Ue=M.next())Ue=oe(z,Ue.value,de),Ue!==null&&(y=o(Ue,y,Pe),Ee===null?_e=Ue:Ee.sibling=Ue,Ee=Ue);return Ze&&sr(z,Pe),_e}for(Ie=s(z,Ie);!Ue.done;Pe++,Ue=M.next())Ue=je(Ie,z,Pe,Ue.value,de),Ue!==null&&(e&&Ue.alternate!==null&&Ie.delete(Ue.key===null?Pe:Ue.key),y=o(Ue,y,Pe),Ee===null?_e=Ue:Ee.sibling=Ue,Ee=Ue);return e&&Ie.forEach(function(yf){return n(z,yf)}),Ze&&sr(z,Pe),_e}function st(z,y,M,de){if(typeof M=="object"&&M!==null&&M.type===h&&M.key===null&&(M=M.props.children),typeof M=="object"&&M!==null){switch(M.$$typeof){case k:e:{for(var _e=M.key,Ee=y;Ee!==null;){if(Ee.key===_e){if(_e=M.type,_e===h){if(Ee.tag===7){r(z,Ee.sibling),y=a(Ee,M.props.children),y.return=z,z=y;break e}}else if(Ee.elementType===_e||typeof _e=="object"&&_e!==null&&_e.$$typeof===ie&&cc(_e)===Ee.type){r(z,Ee.sibling),y=a(Ee,M.props),y.ref=ps(z,Ee,M),y.return=z,z=y;break e}r(z,Ee);break}else n(z,Ee);Ee=Ee.sibling}M.type===h?(y=pr(M.props.children,z.mode,de,M.key),y.return=z,z=y):(de=Ia(M.type,M.key,M.props,null,z.mode,de),de.ref=ps(z,y,M),de.return=z,z=de)}return i(z);case B:e:{for(Ee=M.key;y!==null;){if(y.key===Ee)if(y.tag===4&&y.stateNode.containerInfo===M.containerInfo&&y.stateNode.implementation===M.implementation){r(z,y.sibling),y=a(y,M.children||[]),y.return=z,z=y;break e}else{r(z,y);break}else n(z,y);y=y.sibling}y=kl(M,z.mode,de),y.return=z,z=y}return i(z);case ie:return Ee=M._init,st(z,y,Ee(M._payload),de)}if(We(M))return Se(z,y,M,de);if(X(M))return Ce(z,y,M,de);oa(z,M)}return typeof M=="string"&&M!==""||typeof M=="number"?(M=""+M,y!==null&&y.tag===6?(r(z,y.sibling),y=a(y,M),y.return=z,z=y):(r(z,y),y=wl(M,z.mode,de),y.return=z,z=y),i(z)):r(z,y)}return st}var Pr=dc(!0),uc=dc(!1),la=Fn(null),ia=null,Tr=null,To=null;function Ro(){To=Tr=ia=null}function Mo(e){var n=la.current;Xe(la),e._currentValue=n}function Lo(e,n,r){for(;e!==null;){var s=e.alternate;if((e.childLanes&n)!==n?(e.childLanes|=n,s!==null&&(s.childLanes|=n)):s!==null&&(s.childLanes&n)!==n&&(s.childLanes|=n),e===r)break;e=e.return}}function Rr(e,n){ia=e,To=Tr=null,e=e.dependencies,e!==null&&e.firstContext!==null&&((e.lanes&n)!==0&&(It=!0),e.firstContext=null)}function Bt(e){var n=e._currentValue;if(To!==e)if(e={context:e,memoizedValue:n,next:null},Tr===null){if(ia===null)throw Error(d(308));Tr=e,ia.dependencies={lanes:0,firstContext:e}}else Tr=Tr.next=e;return n}var ar=null;function Fo(e){ar===null?ar=[e]:ar.push(e)}function pc(e,n,r,s){var a=n.interleaved;return a===null?(r.next=r,Fo(n)):(r.next=a.next,a.next=r),n.interleaved=r,yn(e,s)}function yn(e,n){e.lanes|=n;var r=e.alternate;for(r!==null&&(r.lanes|=n),r=e,e=e.return;e!==null;)e.childLanes|=n,r=e.alternate,r!==null&&(r.childLanes|=n),r=e,e=e.return;return r.tag===3?r.stateNode:null}var An=!1;function Do(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,interleaved:null,lanes:0},effects:null}}function fc(e,n){e=e.updateQueue,n.updateQueue===e&&(n.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,effects:e.effects})}function bn(e,n){return{eventTime:e,lane:n,tag:0,payload:null,callback:null,next:null}}function $n(e,n,r){var s=e.updateQueue;if(s===null)return null;if(s=s.shared,($e&2)!==0){var a=s.pending;return a===null?n.next=n:(n.next=a.next,a.next=n),s.pending=n,yn(e,r)}return a=s.interleaved,a===null?(n.next=n,Fo(s)):(n.next=a.next,a.next=n),s.interleaved=n,yn(e,r)}function ca(e,n,r){if(n=n.updateQueue,n!==null&&(n=n.shared,(r&4194240)!==0)){var s=n.lanes;s&=e.pendingLanes,r|=s,n.lanes=r,Xa(e,r)}}function mc(e,n){var r=e.updateQueue,s=e.alternate;if(s!==null&&(s=s.updateQueue,r===s)){var a=null,o=null;if(r=r.firstBaseUpdate,r!==null){do{var i={eventTime:r.eventTime,lane:r.lane,tag:r.tag,payload:r.payload,callback:r.callback,next:null};o===null?a=o=i:o=o.next=i,r=r.next}while(r!==null);o===null?a=o=n:o=o.next=n}else a=o=n;r={baseState:s.baseState,firstBaseUpdate:a,lastBaseUpdate:o,shared:s.shared,effects:s.effects},e.updateQueue=r;return}e=r.lastBaseUpdate,e===null?r.firstBaseUpdate=n:e.next=n,r.lastBaseUpdate=n}function da(e,n,r,s){var a=e.updateQueue;An=!1;var o=a.firstBaseUpdate,i=a.lastBaseUpdate,f=a.shared.pending;if(f!==null){a.shared.pending=null;var g=f,O=g.next;g.next=null,i===null?o=O:i.next=O,i=g;var se=e.alternate;se!==null&&(se=se.updateQueue,f=se.lastBaseUpdate,f!==i&&(f===null?se.firstBaseUpdate=O:f.next=O,se.lastBaseUpdate=g))}if(o!==null){var oe=a.baseState;i=0,se=O=g=null,f=o;do{var ne=f.lane,je=f.eventTime;if((s&ne)===ne){se!==null&&(se=se.next={eventTime:je,lane:0,tag:f.tag,payload:f.payload,callback:f.callback,next:null});e:{var Se=e,Ce=f;switch(ne=n,je=r,Ce.tag){case 1:if(Se=Ce.payload,typeof Se=="function"){oe=Se.call(je,oe,ne);break e}oe=Se;break e;case 3:Se.flags=Se.flags&-65537|128;case 0:if(Se=Ce.payload,ne=typeof Se=="function"?Se.call(je,oe,ne):Se,ne==null)break e;oe=J({},oe,ne);break e;case 2:An=!0}}f.callback!==null&&f.lane!==0&&(e.flags|=64,ne=a.effects,ne===null?a.effects=[f]:ne.push(f))}else je={eventTime:je,lane:ne,tag:f.tag,payload:f.payload,callback:f.callback,next:null},se===null?(O=se=je,g=oe):se=se.next=je,i|=ne;if(f=f.next,f===null){if(f=a.shared.pending,f===null)break;ne=f,f=ne.next,ne.next=null,a.lastBaseUpdate=ne,a.shared.pending=null}}while(!0);if(se===null&&(g=oe),a.baseState=g,a.firstBaseUpdate=O,a.lastBaseUpdate=se,n=a.shared.interleaved,n!==null){a=n;do i|=a.lane,a=a.next;while(a!==n)}else o===null&&(a.shared.lanes=0);ir|=i,e.lanes=i,e.memoizedState=oe}}function xc(e,n,r){if(e=n.effects,n.effects=null,e!==null)for(n=0;n<e.length;n++){var s=e[n],a=s.callback;if(a!==null){if(s.callback=null,s=r,typeof a!="function")throw Error(d(191,a));a.call(s)}}}var fs={},on=Fn(fs),ms=Fn(fs),xs=Fn(fs);function or(e){if(e===fs)throw Error(d(174));return e}function Oo(e,n){switch(qe(xs,n),qe(ms,e),qe(on,fs),e=n.nodeType,e){case 9:case 11:n=(n=n.documentElement)?n.namespaceURI:Mt(null,"");break;default:e=e===8?n.parentNode:n,n=e.namespaceURI||null,e=e.tagName,n=Mt(n,e)}Xe(on),qe(on,n)}function Mr(){Xe(on),Xe(ms),Xe(xs)}function hc(e){or(xs.current);var n=or(on.current),r=Mt(n,e.type);n!==r&&(qe(ms,e),qe(on,r))}function Ao(e){ms.current===e&&(Xe(on),Xe(ms))}var et=Fn(0);function ua(e){for(var n=e;n!==null;){if(n.tag===13){var r=n.memoizedState;if(r!==null&&(r=r.dehydrated,r===null||r.data==="$?"||r.data==="$!"))return n}else if(n.tag===19&&n.memoizedProps.revealOrder!==void 0){if((n.flags&128)!==0)return n}else if(n.child!==null){n.child.return=n,n=n.child;continue}if(n===e)break;for(;n.sibling===null;){if(n.return===null||n.return===e)return null;n=n.return}n.sibling.return=n.return,n=n.sibling}return null}var $o=[];function Uo(){for(var e=0;e<$o.length;e++)$o[e]._workInProgressVersionPrimary=null;$o.length=0}var pa=j.ReactCurrentDispatcher,Vo=j.ReactCurrentBatchConfig,lr=0,tt=null,dt=null,ft=null,fa=!1,hs=!1,gs=0,$p=0;function bt(){throw Error(d(321))}function Bo(e,n){if(n===null)return!1;for(var r=0;r<n.length&&r<e.length;r++)if(!Kt(e[r],n[r]))return!1;return!0}function Wo(e,n,r,s,a,o){if(lr=o,tt=n,n.memoizedState=null,n.updateQueue=null,n.lanes=0,pa.current=e===null||e.memoizedState===null?Wp:Gp,e=r(s,a),hs){o=0;do{if(hs=!1,gs=0,25<=o)throw Error(d(301));o+=1,ft=dt=null,n.updateQueue=null,pa.current=Hp,e=r(s,a)}while(hs)}if(pa.current=ha,n=dt!==null&&dt.next!==null,lr=0,ft=dt=tt=null,fa=!1,n)throw Error(d(300));return e}function Go(){var e=gs!==0;return gs=0,e}function ln(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return ft===null?tt.memoizedState=ft=e:ft=ft.next=e,ft}function Wt(){if(dt===null){var e=tt.alternate;e=e!==null?e.memoizedState:null}else e=dt.next;var n=ft===null?tt.memoizedState:ft.next;if(n!==null)ft=n,dt=e;else{if(e===null)throw Error(d(310));dt=e,e={memoizedState:dt.memoizedState,baseState:dt.baseState,baseQueue:dt.baseQueue,queue:dt.queue,next:null},ft===null?tt.memoizedState=ft=e:ft=ft.next=e}return ft}function vs(e,n){return typeof n=="function"?n(e):n}function Ho(e){var n=Wt(),r=n.queue;if(r===null)throw Error(d(311));r.lastRenderedReducer=e;var s=dt,a=s.baseQueue,o=r.pending;if(o!==null){if(a!==null){var i=a.next;a.next=o.next,o.next=i}s.baseQueue=a=o,r.pending=null}if(a!==null){o=a.next,s=s.baseState;var f=i=null,g=null,O=o;do{var se=O.lane;if((lr&se)===se)g!==null&&(g=g.next={lane:0,action:O.action,hasEagerState:O.hasEagerState,eagerState:O.eagerState,next:null}),s=O.hasEagerState?O.eagerState:e(s,O.action);else{var oe={lane:se,action:O.action,hasEagerState:O.hasEagerState,eagerState:O.eagerState,next:null};g===null?(f=g=oe,i=s):g=g.next=oe,tt.lanes|=se,ir|=se}O=O.next}while(O!==null&&O!==o);g===null?i=s:g.next=f,Kt(s,n.memoizedState)||(It=!0),n.memoizedState=s,n.baseState=i,n.baseQueue=g,r.lastRenderedState=s}if(e=r.interleaved,e!==null){a=e;do o=a.lane,tt.lanes|=o,ir|=o,a=a.next;while(a!==e)}else a===null&&(r.lanes=0);return[n.memoizedState,r.dispatch]}function qo(e){var n=Wt(),r=n.queue;if(r===null)throw Error(d(311));r.lastRenderedReducer=e;var s=r.dispatch,a=r.pending,o=n.memoizedState;if(a!==null){r.pending=null;var i=a=a.next;do o=e(o,i.action),i=i.next;while(i!==a);Kt(o,n.memoizedState)||(It=!0),n.memoizedState=o,n.baseQueue===null&&(n.baseState=o),r.lastRenderedState=o}return[o,s]}function gc(){}function vc(e,n){var r=tt,s=Wt(),a=n(),o=!Kt(s.memoizedState,a);if(o&&(s.memoizedState=a,It=!0),s=s.queue,Qo(jc.bind(null,r,s,e),[e]),s.getSnapshot!==n||o||ft!==null&&ft.memoizedState.tag&1){if(r.flags|=2048,ys(9,bc.bind(null,r,s,a,n),void 0,null),mt===null)throw Error(d(349));(lr&30)!==0||yc(r,n,a)}return a}function yc(e,n,r){e.flags|=16384,e={getSnapshot:n,value:r},n=tt.updateQueue,n===null?(n={lastEffect:null,stores:null},tt.updateQueue=n,n.stores=[e]):(r=n.stores,r===null?n.stores=[e]:r.push(e))}function bc(e,n,r,s){n.value=r,n.getSnapshot=s,wc(n)&&kc(e)}function jc(e,n,r){return r(function(){wc(n)&&kc(e)})}function wc(e){var n=e.getSnapshot;e=e.value;try{var r=n();return!Kt(e,r)}catch{return!0}}function kc(e){var n=yn(e,1);n!==null&&nn(n,e,1,-1)}function Sc(e){var n=ln();return typeof e=="function"&&(e=e()),n.memoizedState=n.baseState=e,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:vs,lastRenderedState:e},n.queue=e,e=e.dispatch=Bp.bind(null,tt,e),[n.memoizedState,e]}function ys(e,n,r,s){return e={tag:e,create:n,destroy:r,deps:s,next:null},n=tt.updateQueue,n===null?(n={lastEffect:null,stores:null},tt.updateQueue=n,n.lastEffect=e.next=e):(r=n.lastEffect,r===null?n.lastEffect=e.next=e:(s=r.next,r.next=e,e.next=s,n.lastEffect=e)),e}function Nc(){return Wt().memoizedState}function ma(e,n,r,s){var a=ln();tt.flags|=e,a.memoizedState=ys(1|n,r,void 0,s===void 0?null:s)}function xa(e,n,r,s){var a=Wt();s=s===void 0?null:s;var o=void 0;if(dt!==null){var i=dt.memoizedState;if(o=i.destroy,s!==null&&Bo(s,i.deps)){a.memoizedState=ys(n,r,o,s);return}}tt.flags|=e,a.memoizedState=ys(1|n,r,o,s)}function Cc(e,n){return ma(8390656,8,e,n)}function Qo(e,n){return xa(2048,8,e,n)}function _c(e,n){return xa(4,2,e,n)}function zc(e,n){return xa(4,4,e,n)}function Ec(e,n){if(typeof n=="function")return e=e(),n(e),function(){n(null)};if(n!=null)return e=e(),n.current=e,function(){n.current=null}}function Ic(e,n,r){return r=r!=null?r.concat([e]):null,xa(4,4,Ec.bind(null,n,e),r)}function Yo(){}function Pc(e,n){var r=Wt();n=n===void 0?null:n;var s=r.memoizedState;return s!==null&&n!==null&&Bo(n,s[1])?s[0]:(r.memoizedState=[e,n],e)}function Tc(e,n){var r=Wt();n=n===void 0?null:n;var s=r.memoizedState;return s!==null&&n!==null&&Bo(n,s[1])?s[0]:(e=e(),r.memoizedState=[e,n],e)}function Rc(e,n,r){return(lr&21)===0?(e.baseState&&(e.baseState=!1,It=!0),e.memoizedState=r):(Kt(r,n)||(r=ci(),tt.lanes|=r,ir|=r,e.baseState=!0),n)}function Up(e,n){var r=Be;Be=r!==0&&4>r?r:4,e(!0);var s=Vo.transition;Vo.transition={};try{e(!1),n()}finally{Be=r,Vo.transition=s}}function Mc(){return Wt().memoizedState}function Vp(e,n,r){var s=Wn(e);if(r={lane:s,action:r,hasEagerState:!1,eagerState:null,next:null},Lc(e))Fc(n,r);else if(r=pc(e,n,r,s),r!==null){var a=Ct();nn(r,e,s,a),Dc(r,n,s)}}function Bp(e,n,r){var s=Wn(e),a={lane:s,action:r,hasEagerState:!1,eagerState:null,next:null};if(Lc(e))Fc(n,a);else{var o=e.alternate;if(e.lanes===0&&(o===null||o.lanes===0)&&(o=n.lastRenderedReducer,o!==null))try{var i=n.lastRenderedState,f=o(i,r);if(a.hasEagerState=!0,a.eagerState=f,Kt(f,i)){var g=n.interleaved;g===null?(a.next=a,Fo(n)):(a.next=g.next,g.next=a),n.interleaved=a;return}}catch{}r=pc(e,n,a,s),r!==null&&(a=Ct(),nn(r,e,s,a),Dc(r,n,s))}}function Lc(e){var n=e.alternate;return e===tt||n!==null&&n===tt}function Fc(e,n){hs=fa=!0;var r=e.pending;r===null?n.next=n:(n.next=r.next,r.next=n),e.pending=n}function Dc(e,n,r){if((r&4194240)!==0){var s=n.lanes;s&=e.pendingLanes,r|=s,n.lanes=r,Xa(e,r)}}var ha={readContext:Bt,useCallback:bt,useContext:bt,useEffect:bt,useImperativeHandle:bt,useInsertionEffect:bt,useLayoutEffect:bt,useMemo:bt,useReducer:bt,useRef:bt,useState:bt,useDebugValue:bt,useDeferredValue:bt,useTransition:bt,useMutableSource:bt,useSyncExternalStore:bt,useId:bt,unstable_isNewReconciler:!1},Wp={readContext:Bt,useCallback:function(e,n){return ln().memoizedState=[e,n===void 0?null:n],e},useContext:Bt,useEffect:Cc,useImperativeHandle:function(e,n,r){return r=r!=null?r.concat([e]):null,ma(4194308,4,Ec.bind(null,n,e),r)},useLayoutEffect:function(e,n){return ma(4194308,4,e,n)},useInsertionEffect:function(e,n){return ma(4,2,e,n)},useMemo:function(e,n){var r=ln();return n=n===void 0?null:n,e=e(),r.memoizedState=[e,n],e},useReducer:function(e,n,r){var s=ln();return n=r!==void 0?r(n):n,s.memoizedState=s.baseState=n,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:n},s.queue=e,e=e.dispatch=Vp.bind(null,tt,e),[s.memoizedState,e]},useRef:function(e){var n=ln();return e={current:e},n.memoizedState=e},useState:Sc,useDebugValue:Yo,useDeferredValue:function(e){return ln().memoizedState=e},useTransition:function(){var e=Sc(!1),n=e[0];return e=Up.bind(null,e[1]),ln().memoizedState=e,[n,e]},useMutableSource:function(){},useSyncExternalStore:function(e,n,r){var s=tt,a=ln();if(Ze){if(r===void 0)throw Error(d(407));r=r()}else{if(r=n(),mt===null)throw Error(d(349));(lr&30)!==0||yc(s,n,r)}a.memoizedState=r;var o={value:r,getSnapshot:n};return a.queue=o,Cc(jc.bind(null,s,o,e),[e]),s.flags|=2048,ys(9,bc.bind(null,s,o,r,n),void 0,null),r},useId:function(){var e=ln(),n=mt.identifierPrefix;if(Ze){var r=vn,s=gn;r=(s&~(1<<32-Xt(s)-1)).toString(32)+r,n=":"+n+"R"+r,r=gs++,0<r&&(n+="H"+r.toString(32)),n+=":"}else r=$p++,n=":"+n+"r"+r.toString(32)+":";return e.memoizedState=n},unstable_isNewReconciler:!1},Gp={readContext:Bt,useCallback:Pc,useContext:Bt,useEffect:Qo,useImperativeHandle:Ic,useInsertionEffect:_c,useLayoutEffect:zc,useMemo:Tc,useReducer:Ho,useRef:Nc,useState:function(){return Ho(vs)},useDebugValue:Yo,useDeferredValue:function(e){var n=Wt();return Rc(n,dt.memoizedState,e)},useTransition:function(){var e=Ho(vs)[0],n=Wt().memoizedState;return[e,n]},useMutableSource:gc,useSyncExternalStore:vc,useId:Mc,unstable_isNewReconciler:!1},Hp={readContext:Bt,useCallback:Pc,useContext:Bt,useEffect:Qo,useImperativeHandle:Ic,useInsertionEffect:_c,useLayoutEffect:zc,useMemo:Tc,useReducer:qo,useRef:Nc,useState:function(){return qo(vs)},useDebugValue:Yo,useDeferredValue:function(e){var n=Wt();return dt===null?n.memoizedState=e:Rc(n,dt.memoizedState,e)},useTransition:function(){var e=qo(vs)[0],n=Wt().memoizedState;return[e,n]},useMutableSource:gc,useSyncExternalStore:vc,useId:Mc,unstable_isNewReconciler:!1};function Zt(e,n){if(e&&e.defaultProps){n=J({},n),e=e.defaultProps;for(var r in e)n[r]===void 0&&(n[r]=e[r]);return n}return n}function Xo(e,n,r,s){n=e.memoizedState,r=r(s,n),r=r==null?n:J({},n,r),e.memoizedState=r,e.lanes===0&&(e.updateQueue.baseState=r)}var ga={isMounted:function(e){return(e=e._reactInternals)?er(e)===e:!1},enqueueSetState:function(e,n,r){e=e._reactInternals;var s=Ct(),a=Wn(e),o=bn(s,a);o.payload=n,r!=null&&(o.callback=r),n=$n(e,o,a),n!==null&&(nn(n,e,a,s),ca(n,e,a))},enqueueReplaceState:function(e,n,r){e=e._reactInternals;var s=Ct(),a=Wn(e),o=bn(s,a);o.tag=1,o.payload=n,r!=null&&(o.callback=r),n=$n(e,o,a),n!==null&&(nn(n,e,a,s),ca(n,e,a))},enqueueForceUpdate:function(e,n){e=e._reactInternals;var r=Ct(),s=Wn(e),a=bn(r,s);a.tag=2,n!=null&&(a.callback=n),n=$n(e,a,s),n!==null&&(nn(n,e,s,r),ca(n,e,s))}};function Oc(e,n,r,s,a,o,i){return e=e.stateNode,typeof e.shouldComponentUpdate=="function"?e.shouldComponentUpdate(s,o,i):n.prototype&&n.prototype.isPureReactComponent?!as(r,s)||!as(a,o):!0}function Ac(e,n,r){var s=!1,a=Dn,o=n.contextType;return typeof o=="object"&&o!==null?o=Bt(o):(a=Et(n)?nr:yt.current,s=n.contextTypes,o=(s=s!=null)?_r(e,a):Dn),n=new n(r,o),e.memoizedState=n.state!==null&&n.state!==void 0?n.state:null,n.updater=ga,e.stateNode=n,n._reactInternals=e,s&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=a,e.__reactInternalMemoizedMaskedChildContext=o),n}function $c(e,n,r,s){e=n.state,typeof n.componentWillReceiveProps=="function"&&n.componentWillReceiveProps(r,s),typeof n.UNSAFE_componentWillReceiveProps=="function"&&n.UNSAFE_componentWillReceiveProps(r,s),n.state!==e&&ga.enqueueReplaceState(n,n.state,null)}function Ko(e,n,r,s){var a=e.stateNode;a.props=r,a.state=e.memoizedState,a.refs={},Do(e);var o=n.contextType;typeof o=="object"&&o!==null?a.context=Bt(o):(o=Et(n)?nr:yt.current,a.context=_r(e,o)),a.state=e.memoizedState,o=n.getDerivedStateFromProps,typeof o=="function"&&(Xo(e,n,o,r),a.state=e.memoizedState),typeof n.getDerivedStateFromProps=="function"||typeof a.getSnapshotBeforeUpdate=="function"||typeof a.UNSAFE_componentWillMount!="function"&&typeof a.componentWillMount!="function"||(n=a.state,typeof a.componentWillMount=="function"&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount=="function"&&a.UNSAFE_componentWillMount(),n!==a.state&&ga.enqueueReplaceState(a,a.state,null),da(e,r,a,s),a.state=e.memoizedState),typeof a.componentDidMount=="function"&&(e.flags|=4194308)}function Lr(e,n){try{var r="",s=n;do r+=F(s),s=s.return;while(s);var a=r}catch(o){a=`
Error generating stack: `+o.message+`
`+o.stack}return{value:e,source:n,stack:a,digest:null}}function Jo(e,n,r){return{value:e,source:null,stack:r??null,digest:n??null}}function Zo(e,n){try{console.error(n.value)}catch(r){setTimeout(function(){throw r})}}var qp=typeof WeakMap=="function"?WeakMap:Map;function Uc(e,n,r){r=bn(-1,r),r.tag=3,r.payload={element:null};var s=n.value;return r.callback=function(){Sa||(Sa=!0,ml=s),Zo(e,n)},r}function Vc(e,n,r){r=bn(-1,r),r.tag=3;var s=e.type.getDerivedStateFromError;if(typeof s=="function"){var a=n.value;r.payload=function(){return s(a)},r.callback=function(){Zo(e,n)}}var o=e.stateNode;return o!==null&&typeof o.componentDidCatch=="function"&&(r.callback=function(){Zo(e,n),typeof s!="function"&&(Vn===null?Vn=new Set([this]):Vn.add(this));var i=n.stack;this.componentDidCatch(n.value,{componentStack:i!==null?i:""})}),r}function Bc(e,n,r){var s=e.pingCache;if(s===null){s=e.pingCache=new qp;var a=new Set;s.set(n,a)}else a=s.get(n),a===void 0&&(a=new Set,s.set(n,a));a.has(r)||(a.add(r),e=lf.bind(null,e,n,r),n.then(e,e))}function Wc(e){do{var n;if((n=e.tag===13)&&(n=e.memoizedState,n=n!==null?n.dehydrated!==null:!0),n)return e;e=e.return}while(e!==null);return null}function Gc(e,n,r,s,a){return(e.mode&1)===0?(e===n?e.flags|=65536:(e.flags|=128,r.flags|=131072,r.flags&=-52805,r.tag===1&&(r.alternate===null?r.tag=17:(n=bn(-1,1),n.tag=2,$n(r,n,1))),r.lanes|=1),e):(e.flags|=65536,e.lanes=a,e)}var Qp=j.ReactCurrentOwner,It=!1;function Nt(e,n,r,s){n.child=e===null?uc(n,null,r,s):Pr(n,e.child,r,s)}function Hc(e,n,r,s,a){r=r.render;var o=n.ref;return Rr(n,a),s=Wo(e,n,r,s,o,a),r=Go(),e!==null&&!It?(n.updateQueue=e.updateQueue,n.flags&=-2053,e.lanes&=~a,jn(e,n,a)):(Ze&&r&&_o(n),n.flags|=1,Nt(e,n,s,a),n.child)}function qc(e,n,r,s,a){if(e===null){var o=r.type;return typeof o=="function"&&!jl(o)&&o.defaultProps===void 0&&r.compare===null&&r.defaultProps===void 0?(n.tag=15,n.type=o,Qc(e,n,o,s,a)):(e=Ia(r.type,null,s,n,n.mode,a),e.ref=n.ref,e.return=n,n.child=e)}if(o=e.child,(e.lanes&a)===0){var i=o.memoizedProps;if(r=r.compare,r=r!==null?r:as,r(i,s)&&e.ref===n.ref)return jn(e,n,a)}return n.flags|=1,e=Hn(o,s),e.ref=n.ref,e.return=n,n.child=e}function Qc(e,n,r,s,a){if(e!==null){var o=e.memoizedProps;if(as(o,s)&&e.ref===n.ref)if(It=!1,n.pendingProps=s=o,(e.lanes&a)!==0)(e.flags&131072)!==0&&(It=!0);else return n.lanes=e.lanes,jn(e,n,a)}return el(e,n,r,s,a)}function Yc(e,n,r){var s=n.pendingProps,a=s.children,o=e!==null?e.memoizedState:null;if(s.mode==="hidden")if((n.mode&1)===0)n.memoizedState={baseLanes:0,cachePool:null,transitions:null},qe(Dr,At),At|=r;else{if((r&1073741824)===0)return e=o!==null?o.baseLanes|r:r,n.lanes=n.childLanes=1073741824,n.memoizedState={baseLanes:e,cachePool:null,transitions:null},n.updateQueue=null,qe(Dr,At),At|=e,null;n.memoizedState={baseLanes:0,cachePool:null,transitions:null},s=o!==null?o.baseLanes:r,qe(Dr,At),At|=s}else o!==null?(s=o.baseLanes|r,n.memoizedState=null):s=r,qe(Dr,At),At|=s;return Nt(e,n,a,r),n.child}function Xc(e,n){var r=n.ref;(e===null&&r!==null||e!==null&&e.ref!==r)&&(n.flags|=512,n.flags|=2097152)}function el(e,n,r,s,a){var o=Et(r)?nr:yt.current;return o=_r(n,o),Rr(n,a),r=Wo(e,n,r,s,o,a),s=Go(),e!==null&&!It?(n.updateQueue=e.updateQueue,n.flags&=-2053,e.lanes&=~a,jn(e,n,a)):(Ze&&s&&_o(n),n.flags|=1,Nt(e,n,r,a),n.child)}function Kc(e,n,r,s,a){if(Et(r)){var o=!0;ta(n)}else o=!1;if(Rr(n,a),n.stateNode===null)ya(e,n),Ac(n,r,s),Ko(n,r,s,a),s=!0;else if(e===null){var i=n.stateNode,f=n.memoizedProps;i.props=f;var g=i.context,O=r.contextType;typeof O=="object"&&O!==null?O=Bt(O):(O=Et(r)?nr:yt.current,O=_r(n,O));var se=r.getDerivedStateFromProps,oe=typeof se=="function"||typeof i.getSnapshotBeforeUpdate=="function";oe||typeof i.UNSAFE_componentWillReceiveProps!="function"&&typeof i.componentWillReceiveProps!="function"||(f!==s||g!==O)&&$c(n,i,s,O),An=!1;var ne=n.memoizedState;i.state=ne,da(n,s,i,a),g=n.memoizedState,f!==s||ne!==g||zt.current||An?(typeof se=="function"&&(Xo(n,r,se,s),g=n.memoizedState),(f=An||Oc(n,r,f,s,ne,g,O))?(oe||typeof i.UNSAFE_componentWillMount!="function"&&typeof i.componentWillMount!="function"||(typeof i.componentWillMount=="function"&&i.componentWillMount(),typeof i.UNSAFE_componentWillMount=="function"&&i.UNSAFE_componentWillMount()),typeof i.componentDidMount=="function"&&(n.flags|=4194308)):(typeof i.componentDidMount=="function"&&(n.flags|=4194308),n.memoizedProps=s,n.memoizedState=g),i.props=s,i.state=g,i.context=O,s=f):(typeof i.componentDidMount=="function"&&(n.flags|=4194308),s=!1)}else{i=n.stateNode,fc(e,n),f=n.memoizedProps,O=n.type===n.elementType?f:Zt(n.type,f),i.props=O,oe=n.pendingProps,ne=i.context,g=r.contextType,typeof g=="object"&&g!==null?g=Bt(g):(g=Et(r)?nr:yt.current,g=_r(n,g));var je=r.getDerivedStateFromProps;(se=typeof je=="function"||typeof i.getSnapshotBeforeUpdate=="function")||typeof i.UNSAFE_componentWillReceiveProps!="function"&&typeof i.componentWillReceiveProps!="function"||(f!==oe||ne!==g)&&$c(n,i,s,g),An=!1,ne=n.memoizedState,i.state=ne,da(n,s,i,a);var Se=n.memoizedState;f!==oe||ne!==Se||zt.current||An?(typeof je=="function"&&(Xo(n,r,je,s),Se=n.memoizedState),(O=An||Oc(n,r,O,s,ne,Se,g)||!1)?(se||typeof i.UNSAFE_componentWillUpdate!="function"&&typeof i.componentWillUpdate!="function"||(typeof i.componentWillUpdate=="function"&&i.componentWillUpdate(s,Se,g),typeof i.UNSAFE_componentWillUpdate=="function"&&i.UNSAFE_componentWillUpdate(s,Se,g)),typeof i.componentDidUpdate=="function"&&(n.flags|=4),typeof i.getSnapshotBeforeUpdate=="function"&&(n.flags|=1024)):(typeof i.componentDidUpdate!="function"||f===e.memoizedProps&&ne===e.memoizedState||(n.flags|=4),typeof i.getSnapshotBeforeUpdate!="function"||f===e.memoizedProps&&ne===e.memoizedState||(n.flags|=1024),n.memoizedProps=s,n.memoizedState=Se),i.props=s,i.state=Se,i.context=g,s=O):(typeof i.componentDidUpdate!="function"||f===e.memoizedProps&&ne===e.memoizedState||(n.flags|=4),typeof i.getSnapshotBeforeUpdate!="function"||f===e.memoizedProps&&ne===e.memoizedState||(n.flags|=1024),s=!1)}return tl(e,n,r,s,o,a)}function tl(e,n,r,s,a,o){Xc(e,n);var i=(n.flags&128)!==0;if(!s&&!i)return a&&nc(n,r,!1),jn(e,n,o);s=n.stateNode,Qp.current=n;var f=i&&typeof r.getDerivedStateFromError!="function"?null:s.render();return n.flags|=1,e!==null&&i?(n.child=Pr(n,e.child,null,o),n.child=Pr(n,null,f,o)):Nt(e,n,f,o),n.memoizedState=s.state,a&&nc(n,r,!0),n.child}function Jc(e){var n=e.stateNode;n.pendingContext?ec(e,n.pendingContext,n.pendingContext!==n.context):n.context&&ec(e,n.context,!1),Oo(e,n.containerInfo)}function Zc(e,n,r,s,a){return Ir(),Po(a),n.flags|=256,Nt(e,n,r,s),n.child}var nl={dehydrated:null,treeContext:null,retryLane:0};function rl(e){return{baseLanes:e,cachePool:null,transitions:null}}function ed(e,n,r){var s=n.pendingProps,a=et.current,o=!1,i=(n.flags&128)!==0,f;if((f=i)||(f=e!==null&&e.memoizedState===null?!1:(a&2)!==0),f?(o=!0,n.flags&=-129):(e===null||e.memoizedState!==null)&&(a|=1),qe(et,a&1),e===null)return Io(n),e=n.memoizedState,e!==null&&(e=e.dehydrated,e!==null)?((n.mode&1)===0?n.lanes=1:e.data==="$!"?n.lanes=8:n.lanes=1073741824,null):(i=s.children,e=s.fallback,o?(s=n.mode,o=n.child,i={mode:"hidden",children:i},(s&1)===0&&o!==null?(o.childLanes=0,o.pendingProps=i):o=Pa(i,s,0,null),e=pr(e,s,r,null),o.return=n,e.return=n,o.sibling=e,n.child=o,n.child.memoizedState=rl(r),n.memoizedState=nl,e):sl(n,i));if(a=e.memoizedState,a!==null&&(f=a.dehydrated,f!==null))return Yp(e,n,i,s,f,a,r);if(o){o=s.fallback,i=n.mode,a=e.child,f=a.sibling;var g={mode:"hidden",children:s.children};return(i&1)===0&&n.child!==a?(s=n.child,s.childLanes=0,s.pendingProps=g,n.deletions=null):(s=Hn(a,g),s.subtreeFlags=a.subtreeFlags&14680064),f!==null?o=Hn(f,o):(o=pr(o,i,r,null),o.flags|=2),o.return=n,s.return=n,s.sibling=o,n.child=s,s=o,o=n.child,i=e.child.memoizedState,i=i===null?rl(r):{baseLanes:i.baseLanes|r,cachePool:null,transitions:i.transitions},o.memoizedState=i,o.childLanes=e.childLanes&~r,n.memoizedState=nl,s}return o=e.child,e=o.sibling,s=Hn(o,{mode:"visible",children:s.children}),(n.mode&1)===0&&(s.lanes=r),s.return=n,s.sibling=null,e!==null&&(r=n.deletions,r===null?(n.deletions=[e],n.flags|=16):r.push(e)),n.child=s,n.memoizedState=null,s}function sl(e,n){return n=Pa({mode:"visible",children:n},e.mode,0,null),n.return=e,e.child=n}function va(e,n,r,s){return s!==null&&Po(s),Pr(n,e.child,null,r),e=sl(n,n.pendingProps.children),e.flags|=2,n.memoizedState=null,e}function Yp(e,n,r,s,a,o,i){if(r)return n.flags&256?(n.flags&=-257,s=Jo(Error(d(422))),va(e,n,i,s)):n.memoizedState!==null?(n.child=e.child,n.flags|=128,null):(o=s.fallback,a=n.mode,s=Pa({mode:"visible",children:s.children},a,0,null),o=pr(o,a,i,null),o.flags|=2,s.return=n,o.return=n,s.sibling=o,n.child=s,(n.mode&1)!==0&&Pr(n,e.child,null,i),n.child.memoizedState=rl(i),n.memoizedState=nl,o);if((n.mode&1)===0)return va(e,n,i,null);if(a.data==="$!"){if(s=a.nextSibling&&a.nextSibling.dataset,s)var f=s.dgst;return s=f,o=Error(d(419)),s=Jo(o,s,void 0),va(e,n,i,s)}if(f=(i&e.childLanes)!==0,It||f){if(s=mt,s!==null){switch(i&-i){case 4:a=2;break;case 16:a=8;break;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:a=32;break;case 536870912:a=268435456;break;default:a=0}a=(a&(s.suspendedLanes|i))!==0?0:a,a!==0&&a!==o.retryLane&&(o.retryLane=a,yn(e,a),nn(s,e,a,-1))}return bl(),s=Jo(Error(d(421))),va(e,n,i,s)}return a.data==="$?"?(n.flags|=128,n.child=e.child,n=cf.bind(null,e),a._reactRetry=n,null):(e=o.treeContext,Ot=Ln(a.nextSibling),Dt=n,Ze=!0,Jt=null,e!==null&&(Ut[Vt++]=gn,Ut[Vt++]=vn,Ut[Vt++]=rr,gn=e.id,vn=e.overflow,rr=n),n=sl(n,s.children),n.flags|=4096,n)}function td(e,n,r){e.lanes|=n;var s=e.alternate;s!==null&&(s.lanes|=n),Lo(e.return,n,r)}function al(e,n,r,s,a){var o=e.memoizedState;o===null?e.memoizedState={isBackwards:n,rendering:null,renderingStartTime:0,last:s,tail:r,tailMode:a}:(o.isBackwards=n,o.rendering=null,o.renderingStartTime=0,o.last=s,o.tail=r,o.tailMode=a)}function nd(e,n,r){var s=n.pendingProps,a=s.revealOrder,o=s.tail;if(Nt(e,n,s.children,r),s=et.current,(s&2)!==0)s=s&1|2,n.flags|=128;else{if(e!==null&&(e.flags&128)!==0)e:for(e=n.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&td(e,r,n);else if(e.tag===19)td(e,r,n);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===n)break e;for(;e.sibling===null;){if(e.return===null||e.return===n)break e;e=e.return}e.sibling.return=e.return,e=e.sibling}s&=1}if(qe(et,s),(n.mode&1)===0)n.memoizedState=null;else switch(a){case"forwards":for(r=n.child,a=null;r!==null;)e=r.alternate,e!==null&&ua(e)===null&&(a=r),r=r.sibling;r=a,r===null?(a=n.child,n.child=null):(a=r.sibling,r.sibling=null),al(n,!1,a,r,o);break;case"backwards":for(r=null,a=n.child,n.child=null;a!==null;){if(e=a.alternate,e!==null&&ua(e)===null){n.child=a;break}e=a.sibling,a.sibling=r,r=a,a=e}al(n,!0,r,null,o);break;case"together":al(n,!1,null,null,void 0);break;default:n.memoizedState=null}return n.child}function ya(e,n){(n.mode&1)===0&&e!==null&&(e.alternate=null,n.alternate=null,n.flags|=2)}function jn(e,n,r){if(e!==null&&(n.dependencies=e.dependencies),ir|=n.lanes,(r&n.childLanes)===0)return null;if(e!==null&&n.child!==e.child)throw Error(d(153));if(n.child!==null){for(e=n.child,r=Hn(e,e.pendingProps),n.child=r,r.return=n;e.sibling!==null;)e=e.sibling,r=r.sibling=Hn(e,e.pendingProps),r.return=n;r.sibling=null}return n.child}function Xp(e,n,r){switch(n.tag){case 3:Jc(n),Ir();break;case 5:hc(n);break;case 1:Et(n.type)&&ta(n);break;case 4:Oo(n,n.stateNode.containerInfo);break;case 10:var s=n.type._context,a=n.memoizedProps.value;qe(la,s._currentValue),s._currentValue=a;break;case 13:if(s=n.memoizedState,s!==null)return s.dehydrated!==null?(qe(et,et.current&1),n.flags|=128,null):(r&n.child.childLanes)!==0?ed(e,n,r):(qe(et,et.current&1),e=jn(e,n,r),e!==null?e.sibling:null);qe(et,et.current&1);break;case 19:if(s=(r&n.childLanes)!==0,(e.flags&128)!==0){if(s)return nd(e,n,r);n.flags|=128}if(a=n.memoizedState,a!==null&&(a.rendering=null,a.tail=null,a.lastEffect=null),qe(et,et.current),s)break;return null;case 22:case 23:return n.lanes=0,Yc(e,n,r)}return jn(e,n,r)}var rd,ol,sd,ad;rd=function(e,n){for(var r=n.child;r!==null;){if(r.tag===5||r.tag===6)e.appendChild(r.stateNode);else if(r.tag!==4&&r.child!==null){r.child.return=r,r=r.child;continue}if(r===n)break;for(;r.sibling===null;){if(r.return===null||r.return===n)return;r=r.return}r.sibling.return=r.return,r=r.sibling}},ol=function(){},sd=function(e,n,r,s){var a=e.memoizedProps;if(a!==s){e=n.stateNode,or(on.current);var o=null;switch(r){case"input":a=Re(e,a),s=Re(e,s),o=[];break;case"select":a=J({},a,{value:void 0}),s=J({},s,{value:void 0}),o=[];break;case"textarea":a=_n(e,a),s=_n(e,s),o=[];break;default:typeof a.onClick!="function"&&typeof s.onClick=="function"&&(e.onclick=Js)}ke(r,s);var i;r=null;for(O in a)if(!s.hasOwnProperty(O)&&a.hasOwnProperty(O)&&a[O]!=null)if(O==="style"){var f=a[O];for(i in f)f.hasOwnProperty(i)&&(r||(r={}),r[i]="")}else O!=="dangerouslySetInnerHTML"&&O!=="children"&&O!=="suppressContentEditableWarning"&&O!=="suppressHydrationWarning"&&O!=="autoFocus"&&(p.hasOwnProperty(O)?o||(o=[]):(o=o||[]).push(O,null));for(O in s){var g=s[O];if(f=a?.[O],s.hasOwnProperty(O)&&g!==f&&(g!=null||f!=null))if(O==="style")if(f){for(i in f)!f.hasOwnProperty(i)||g&&g.hasOwnProperty(i)||(r||(r={}),r[i]="");for(i in g)g.hasOwnProperty(i)&&f[i]!==g[i]&&(r||(r={}),r[i]=g[i])}else r||(o||(o=[]),o.push(O,r)),r=g;else O==="dangerouslySetInnerHTML"?(g=g?g.__html:void 0,f=f?f.__html:void 0,g!=null&&f!==g&&(o=o||[]).push(O,g)):O==="children"?typeof g!="string"&&typeof g!="number"||(o=o||[]).push(O,""+g):O!=="suppressContentEditableWarning"&&O!=="suppressHydrationWarning"&&(p.hasOwnProperty(O)?(g!=null&&O==="onScroll"&&Ye("scroll",e),o||f===g||(o=[])):(o=o||[]).push(O,g))}r&&(o=o||[]).push("style",r);var O=o;(n.updateQueue=O)&&(n.flags|=4)}},ad=function(e,n,r,s){r!==s&&(n.flags|=4)};function bs(e,n){if(!Ze)switch(e.tailMode){case"hidden":n=e.tail;for(var r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?e.tail=null:r.sibling=null;break;case"collapsed":r=e.tail;for(var s=null;r!==null;)r.alternate!==null&&(s=r),r=r.sibling;s===null?n||e.tail===null?e.tail=null:e.tail.sibling=null:s.sibling=null}}function jt(e){var n=e.alternate!==null&&e.alternate.child===e.child,r=0,s=0;if(n)for(var a=e.child;a!==null;)r|=a.lanes|a.childLanes,s|=a.subtreeFlags&14680064,s|=a.flags&14680064,a.return=e,a=a.sibling;else for(a=e.child;a!==null;)r|=a.lanes|a.childLanes,s|=a.subtreeFlags,s|=a.flags,a.return=e,a=a.sibling;return e.subtreeFlags|=s,e.childLanes=r,n}function Kp(e,n,r){var s=n.pendingProps;switch(zo(n),n.tag){case 2:case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return jt(n),null;case 1:return Et(n.type)&&ea(),jt(n),null;case 3:return s=n.stateNode,Mr(),Xe(zt),Xe(yt),Uo(),s.pendingContext&&(s.context=s.pendingContext,s.pendingContext=null),(e===null||e.child===null)&&(aa(n)?n.flags|=4:e===null||e.memoizedState.isDehydrated&&(n.flags&256)===0||(n.flags|=1024,Jt!==null&&(gl(Jt),Jt=null))),ol(e,n),jt(n),null;case 5:Ao(n);var a=or(xs.current);if(r=n.type,e!==null&&n.stateNode!=null)sd(e,n,r,s,a),e.ref!==n.ref&&(n.flags|=512,n.flags|=2097152);else{if(!s){if(n.stateNode===null)throw Error(d(166));return jt(n),null}if(e=or(on.current),aa(n)){s=n.stateNode,r=n.type;var o=n.memoizedProps;switch(s[an]=n,s[ds]=o,e=(n.mode&1)!==0,r){case"dialog":Ye("cancel",s),Ye("close",s);break;case"iframe":case"object":case"embed":Ye("load",s);break;case"video":case"audio":for(a=0;a<ls.length;a++)Ye(ls[a],s);break;case"source":Ye("error",s);break;case"img":case"image":case"link":Ye("error",s),Ye("load",s);break;case"details":Ye("toggle",s);break;case"input":Ve(s,o),Ye("invalid",s);break;case"select":s._wrapperState={wasMultiple:!!o.multiple},Ye("invalid",s);break;case"textarea":pn(s,o),Ye("invalid",s)}ke(r,o),a=null;for(var i in o)if(o.hasOwnProperty(i)){var f=o[i];i==="children"?typeof f=="string"?s.textContent!==f&&(o.suppressHydrationWarning!==!0&&Ks(s.textContent,f,e),a=["children",f]):typeof f=="number"&&s.textContent!==""+f&&(o.suppressHydrationWarning!==!0&&Ks(s.textContent,f,e),a=["children",""+f]):p.hasOwnProperty(i)&&f!=null&&i==="onScroll"&&Ye("scroll",s)}switch(r){case"input":ze(s),Kn(s,o,!0);break;case"textarea":ze(s),zn(s);break;case"select":case"option":break;default:typeof o.onClick=="function"&&(s.onclick=Js)}s=a,n.updateQueue=s,s!==null&&(n.flags|=4)}else{i=a.nodeType===9?a:a.ownerDocument,e==="http://www.w3.org/1999/xhtml"&&(e=gr(r)),e==="http://www.w3.org/1999/xhtml"?r==="script"?(e=i.createElement("div"),e.innerHTML="<script><\/script>",e=e.removeChild(e.firstChild)):typeof s.is=="string"?e=i.createElement(r,{is:s.is}):(e=i.createElement(r),r==="select"&&(i=e,s.multiple?i.multiple=!0:s.size&&(i.size=s.size))):e=i.createElementNS(e,r),e[an]=n,e[ds]=s,rd(e,n,!1,!1),n.stateNode=e;e:{switch(i=Ae(r,s),r){case"dialog":Ye("cancel",e),Ye("close",e),a=s;break;case"iframe":case"object":case"embed":Ye("load",e),a=s;break;case"video":case"audio":for(a=0;a<ls.length;a++)Ye(ls[a],e);a=s;break;case"source":Ye("error",e),a=s;break;case"img":case"image":case"link":Ye("error",e),Ye("load",e),a=s;break;case"details":Ye("toggle",e),a=s;break;case"input":Ve(e,s),a=Re(e,s),Ye("invalid",e);break;case"option":a=s;break;case"select":e._wrapperState={wasMultiple:!!s.multiple},a=J({},s,{value:void 0}),Ye("invalid",e);break;case"textarea":pn(e,s),a=_n(e,s),Ye("invalid",e);break;default:a=s}ke(r,a),f=a;for(o in f)if(f.hasOwnProperty(o)){var g=f[o];o==="style"?ce(e,g):o==="dangerouslySetInnerHTML"?(g=g?g.__html:void 0,g!=null&&mn(e,g)):o==="children"?typeof g=="string"?(r!=="textarea"||g!=="")&&$t(e,g):typeof g=="number"&&$t(e,""+g):o!=="suppressContentEditableWarning"&&o!=="suppressHydrationWarning"&&o!=="autoFocus"&&(p.hasOwnProperty(o)?g!=null&&o==="onScroll"&&Ye("scroll",e):g!=null&&K(e,o,g,i))}switch(r){case"input":ze(e),Kn(e,s,!1);break;case"textarea":ze(e),zn(e);break;case"option":s.value!=null&&e.setAttribute("value",""+Q(s.value));break;case"select":e.multiple=!!s.multiple,o=s.value,o!=null?kt(e,!!s.multiple,o,!1):s.defaultValue!=null&&kt(e,!!s.multiple,s.defaultValue,!0);break;default:typeof a.onClick=="function"&&(e.onclick=Js)}switch(r){case"button":case"input":case"select":case"textarea":s=!!s.autoFocus;break e;case"img":s=!0;break e;default:s=!1}}s&&(n.flags|=4)}n.ref!==null&&(n.flags|=512,n.flags|=2097152)}return jt(n),null;case 6:if(e&&n.stateNode!=null)ad(e,n,e.memoizedProps,s);else{if(typeof s!="string"&&n.stateNode===null)throw Error(d(166));if(r=or(xs.current),or(on.current),aa(n)){if(s=n.stateNode,r=n.memoizedProps,s[an]=n,(o=s.nodeValue!==r)&&(e=Dt,e!==null))switch(e.tag){case 3:Ks(s.nodeValue,r,(e.mode&1)!==0);break;case 5:e.memoizedProps.suppressHydrationWarning!==!0&&Ks(s.nodeValue,r,(e.mode&1)!==0)}o&&(n.flags|=4)}else s=(r.nodeType===9?r:r.ownerDocument).createTextNode(s),s[an]=n,n.stateNode=s}return jt(n),null;case 13:if(Xe(et),s=n.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(Ze&&Ot!==null&&(n.mode&1)!==0&&(n.flags&128)===0)ic(),Ir(),n.flags|=98560,o=!1;else if(o=aa(n),s!==null&&s.dehydrated!==null){if(e===null){if(!o)throw Error(d(318));if(o=n.memoizedState,o=o!==null?o.dehydrated:null,!o)throw Error(d(317));o[an]=n}else Ir(),(n.flags&128)===0&&(n.memoizedState=null),n.flags|=4;jt(n),o=!1}else Jt!==null&&(gl(Jt),Jt=null),o=!0;if(!o)return n.flags&65536?n:null}return(n.flags&128)!==0?(n.lanes=r,n):(s=s!==null,s!==(e!==null&&e.memoizedState!==null)&&s&&(n.child.flags|=8192,(n.mode&1)!==0&&(e===null||(et.current&1)!==0?ut===0&&(ut=3):bl())),n.updateQueue!==null&&(n.flags|=4),jt(n),null);case 4:return Mr(),ol(e,n),e===null&&is(n.stateNode.containerInfo),jt(n),null;case 10:return Mo(n.type._context),jt(n),null;case 17:return Et(n.type)&&ea(),jt(n),null;case 19:if(Xe(et),o=n.memoizedState,o===null)return jt(n),null;if(s=(n.flags&128)!==0,i=o.rendering,i===null)if(s)bs(o,!1);else{if(ut!==0||e!==null&&(e.flags&128)!==0)for(e=n.child;e!==null;){if(i=ua(e),i!==null){for(n.flags|=128,bs(o,!1),s=i.updateQueue,s!==null&&(n.updateQueue=s,n.flags|=4),n.subtreeFlags=0,s=r,r=n.child;r!==null;)o=r,e=s,o.flags&=14680066,i=o.alternate,i===null?(o.childLanes=0,o.lanes=e,o.child=null,o.subtreeFlags=0,o.memoizedProps=null,o.memoizedState=null,o.updateQueue=null,o.dependencies=null,o.stateNode=null):(o.childLanes=i.childLanes,o.lanes=i.lanes,o.child=i.child,o.subtreeFlags=0,o.deletions=null,o.memoizedProps=i.memoizedProps,o.memoizedState=i.memoizedState,o.updateQueue=i.updateQueue,o.type=i.type,e=i.dependencies,o.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),r=r.sibling;return qe(et,et.current&1|2),n.child}e=e.sibling}o.tail!==null&&rt()>Or&&(n.flags|=128,s=!0,bs(o,!1),n.lanes=4194304)}else{if(!s)if(e=ua(i),e!==null){if(n.flags|=128,s=!0,r=e.updateQueue,r!==null&&(n.updateQueue=r,n.flags|=4),bs(o,!0),o.tail===null&&o.tailMode==="hidden"&&!i.alternate&&!Ze)return jt(n),null}else 2*rt()-o.renderingStartTime>Or&&r!==1073741824&&(n.flags|=128,s=!0,bs(o,!1),n.lanes=4194304);o.isBackwards?(i.sibling=n.child,n.child=i):(r=o.last,r!==null?r.sibling=i:n.child=i,o.last=i)}return o.tail!==null?(n=o.tail,o.rendering=n,o.tail=n.sibling,o.renderingStartTime=rt(),n.sibling=null,r=et.current,qe(et,s?r&1|2:r&1),n):(jt(n),null);case 22:case 23:return yl(),s=n.memoizedState!==null,e!==null&&e.memoizedState!==null!==s&&(n.flags|=8192),s&&(n.mode&1)!==0?(At&1073741824)!==0&&(jt(n),n.subtreeFlags&6&&(n.flags|=8192)):jt(n),null;case 24:return null;case 25:return null}throw Error(d(156,n.tag))}function Jp(e,n){switch(zo(n),n.tag){case 1:return Et(n.type)&&ea(),e=n.flags,e&65536?(n.flags=e&-65537|128,n):null;case 3:return Mr(),Xe(zt),Xe(yt),Uo(),e=n.flags,(e&65536)!==0&&(e&128)===0?(n.flags=e&-65537|128,n):null;case 5:return Ao(n),null;case 13:if(Xe(et),e=n.memoizedState,e!==null&&e.dehydrated!==null){if(n.alternate===null)throw Error(d(340));Ir()}return e=n.flags,e&65536?(n.flags=e&-65537|128,n):null;case 19:return Xe(et),null;case 4:return Mr(),null;case 10:return Mo(n.type._context),null;case 22:case 23:return yl(),null;case 24:return null;default:return null}}var ba=!1,wt=!1,Zp=typeof WeakSet=="function"?WeakSet:Set,we=null;function Fr(e,n){var r=e.ref;if(r!==null)if(typeof r=="function")try{r(null)}catch(s){nt(e,n,s)}else r.current=null}function ll(e,n,r){try{r()}catch(s){nt(e,n,s)}}var od=!1;function ef(e,n){if(yo=$s,e=Oi(),uo(e)){if("selectionStart"in e)var r={start:e.selectionStart,end:e.selectionEnd};else e:{r=(r=e.ownerDocument)&&r.defaultView||window;var s=r.getSelection&&r.getSelection();if(s&&s.rangeCount!==0){r=s.anchorNode;var a=s.anchorOffset,o=s.focusNode;s=s.focusOffset;try{r.nodeType,o.nodeType}catch{r=null;break e}var i=0,f=-1,g=-1,O=0,se=0,oe=e,ne=null;t:for(;;){for(var je;oe!==r||a!==0&&oe.nodeType!==3||(f=i+a),oe!==o||s!==0&&oe.nodeType!==3||(g=i+s),oe.nodeType===3&&(i+=oe.nodeValue.length),(je=oe.firstChild)!==null;)ne=oe,oe=je;for(;;){if(oe===e)break t;if(ne===r&&++O===a&&(f=i),ne===o&&++se===s&&(g=i),(je=oe.nextSibling)!==null)break;oe=ne,ne=oe.parentNode}oe=je}r=f===-1||g===-1?null:{start:f,end:g}}else r=null}r=r||{start:0,end:0}}else r=null;for(bo={focusedElem:e,selectionRange:r},$s=!1,we=n;we!==null;)if(n=we,e=n.child,(n.subtreeFlags&1028)!==0&&e!==null)e.return=n,we=e;else for(;we!==null;){n=we;try{var Se=n.alternate;if((n.flags&1024)!==0)switch(n.tag){case 0:case 11:case 15:break;case 1:if(Se!==null){var Ce=Se.memoizedProps,st=Se.memoizedState,z=n.stateNode,y=z.getSnapshotBeforeUpdate(n.elementType===n.type?Ce:Zt(n.type,Ce),st);z.__reactInternalSnapshotBeforeUpdate=y}break;case 3:var M=n.stateNode.containerInfo;M.nodeType===1?M.textContent="":M.nodeType===9&&M.documentElement&&M.removeChild(M.documentElement);break;case 5:case 6:case 4:case 17:break;default:throw Error(d(163))}}catch(de){nt(n,n.return,de)}if(e=n.sibling,e!==null){e.return=n.return,we=e;break}we=n.return}return Se=od,od=!1,Se}function js(e,n,r){var s=n.updateQueue;if(s=s!==null?s.lastEffect:null,s!==null){var a=s=s.next;do{if((a.tag&e)===e){var o=a.destroy;a.destroy=void 0,o!==void 0&&ll(n,r,o)}a=a.next}while(a!==s)}}function ja(e,n){if(n=n.updateQueue,n=n!==null?n.lastEffect:null,n!==null){var r=n=n.next;do{if((r.tag&e)===e){var s=r.create;r.destroy=s()}r=r.next}while(r!==n)}}function il(e){var n=e.ref;if(n!==null){var r=e.stateNode;e.tag,e=r,typeof n=="function"?n(e):n.current=e}}function ld(e){var n=e.alternate;n!==null&&(e.alternate=null,ld(n)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(n=e.stateNode,n!==null&&(delete n[an],delete n[ds],delete n[So],delete n[Fp],delete n[Dp])),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}function id(e){return e.tag===5||e.tag===3||e.tag===4}function cd(e){e:for(;;){for(;e.sibling===null;){if(e.return===null||id(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.flags&2||e.child===null||e.tag===4)continue e;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function cl(e,n,r){var s=e.tag;if(s===5||s===6)e=e.stateNode,n?r.nodeType===8?r.parentNode.insertBefore(e,n):r.insertBefore(e,n):(r.nodeType===8?(n=r.parentNode,n.insertBefore(e,r)):(n=r,n.appendChild(e)),r=r._reactRootContainer,r!=null||n.onclick!==null||(n.onclick=Js));else if(s!==4&&(e=e.child,e!==null))for(cl(e,n,r),e=e.sibling;e!==null;)cl(e,n,r),e=e.sibling}function dl(e,n,r){var s=e.tag;if(s===5||s===6)e=e.stateNode,n?r.insertBefore(e,n):r.appendChild(e);else if(s!==4&&(e=e.child,e!==null))for(dl(e,n,r),e=e.sibling;e!==null;)dl(e,n,r),e=e.sibling}var gt=null,en=!1;function Un(e,n,r){for(r=r.child;r!==null;)dd(e,n,r),r=r.sibling}function dd(e,n,r){if(sn&&typeof sn.onCommitFiberUnmount=="function")try{sn.onCommitFiberUnmount(Ms,r)}catch{}switch(r.tag){case 5:wt||Fr(r,n);case 6:var s=gt,a=en;gt=null,Un(e,n,r),gt=s,en=a,gt!==null&&(en?(e=gt,r=r.stateNode,e.nodeType===8?e.parentNode.removeChild(r):e.removeChild(r)):gt.removeChild(r.stateNode));break;case 18:gt!==null&&(en?(e=gt,r=r.stateNode,e.nodeType===8?ko(e.parentNode,r):e.nodeType===1&&ko(e,r),Zr(e)):ko(gt,r.stateNode));break;case 4:s=gt,a=en,gt=r.stateNode.containerInfo,en=!0,Un(e,n,r),gt=s,en=a;break;case 0:case 11:case 14:case 15:if(!wt&&(s=r.updateQueue,s!==null&&(s=s.lastEffect,s!==null))){a=s=s.next;do{var o=a,i=o.destroy;o=o.tag,i!==void 0&&((o&2)!==0||(o&4)!==0)&&ll(r,n,i),a=a.next}while(a!==s)}Un(e,n,r);break;case 1:if(!wt&&(Fr(r,n),s=r.stateNode,typeof s.componentWillUnmount=="function"))try{s.props=r.memoizedProps,s.state=r.memoizedState,s.componentWillUnmount()}catch(f){nt(r,n,f)}Un(e,n,r);break;case 21:Un(e,n,r);break;case 22:r.mode&1?(wt=(s=wt)||r.memoizedState!==null,Un(e,n,r),wt=s):Un(e,n,r);break;default:Un(e,n,r)}}function ud(e){var n=e.updateQueue;if(n!==null){e.updateQueue=null;var r=e.stateNode;r===null&&(r=e.stateNode=new Zp),n.forEach(function(s){var a=df.bind(null,e,s);r.has(s)||(r.add(s),s.then(a,a))})}}function tn(e,n){var r=n.deletions;if(r!==null)for(var s=0;s<r.length;s++){var a=r[s];try{var o=e,i=n,f=i;e:for(;f!==null;){switch(f.tag){case 5:gt=f.stateNode,en=!1;break e;case 3:gt=f.stateNode.containerInfo,en=!0;break e;case 4:gt=f.stateNode.containerInfo,en=!0;break e}f=f.return}if(gt===null)throw Error(d(160));dd(o,i,a),gt=null,en=!1;var g=a.alternate;g!==null&&(g.return=null),a.return=null}catch(O){nt(a,n,O)}}if(n.subtreeFlags&12854)for(n=n.child;n!==null;)pd(n,e),n=n.sibling}function pd(e,n){var r=e.alternate,s=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(tn(n,e),cn(e),s&4){try{js(3,e,e.return),ja(3,e)}catch(Ce){nt(e,e.return,Ce)}try{js(5,e,e.return)}catch(Ce){nt(e,e.return,Ce)}}break;case 1:tn(n,e),cn(e),s&512&&r!==null&&Fr(r,r.return);break;case 5:if(tn(n,e),cn(e),s&512&&r!==null&&Fr(r,r.return),e.flags&32){var a=e.stateNode;try{$t(a,"")}catch(Ce){nt(e,e.return,Ce)}}if(s&4&&(a=e.stateNode,a!=null)){var o=e.memoizedProps,i=r!==null?r.memoizedProps:o,f=e.type,g=e.updateQueue;if(e.updateQueue=null,g!==null)try{f==="input"&&o.type==="radio"&&o.name!=null&&it(a,o),Ae(f,i);var O=Ae(f,o);for(i=0;i<g.length;i+=2){var se=g[i],oe=g[i+1];se==="style"?ce(a,oe):se==="dangerouslySetInnerHTML"?mn(a,oe):se==="children"?$t(a,oe):K(a,se,oe,O)}switch(f){case"input":Je(a,o);break;case"textarea":hr(a,o);break;case"select":var ne=a._wrapperState.wasMultiple;a._wrapperState.wasMultiple=!!o.multiple;var je=o.value;je!=null?kt(a,!!o.multiple,je,!1):ne!==!!o.multiple&&(o.defaultValue!=null?kt(a,!!o.multiple,o.defaultValue,!0):kt(a,!!o.multiple,o.multiple?[]:"",!1))}a[ds]=o}catch(Ce){nt(e,e.return,Ce)}}break;case 6:if(tn(n,e),cn(e),s&4){if(e.stateNode===null)throw Error(d(162));a=e.stateNode,o=e.memoizedProps;try{a.nodeValue=o}catch(Ce){nt(e,e.return,Ce)}}break;case 3:if(tn(n,e),cn(e),s&4&&r!==null&&r.memoizedState.isDehydrated)try{Zr(n.containerInfo)}catch(Ce){nt(e,e.return,Ce)}break;case 4:tn(n,e),cn(e);break;case 13:tn(n,e),cn(e),a=e.child,a.flags&8192&&(o=a.memoizedState!==null,a.stateNode.isHidden=o,!o||a.alternate!==null&&a.alternate.memoizedState!==null||(fl=rt())),s&4&&ud(e);break;case 22:if(se=r!==null&&r.memoizedState!==null,e.mode&1?(wt=(O=wt)||se,tn(n,e),wt=O):tn(n,e),cn(e),s&8192){if(O=e.memoizedState!==null,(e.stateNode.isHidden=O)&&!se&&(e.mode&1)!==0)for(we=e,se=e.child;se!==null;){for(oe=we=se;we!==null;){switch(ne=we,je=ne.child,ne.tag){case 0:case 11:case 14:case 15:js(4,ne,ne.return);break;case 1:Fr(ne,ne.return);var Se=ne.stateNode;if(typeof Se.componentWillUnmount=="function"){s=ne,r=ne.return;try{n=s,Se.props=n.memoizedProps,Se.state=n.memoizedState,Se.componentWillUnmount()}catch(Ce){nt(s,r,Ce)}}break;case 5:Fr(ne,ne.return);break;case 22:if(ne.memoizedState!==null){xd(oe);continue}}je!==null?(je.return=ne,we=je):xd(oe)}se=se.sibling}e:for(se=null,oe=e;;){if(oe.tag===5){if(se===null){se=oe;try{a=oe.stateNode,O?(o=a.style,typeof o.setProperty=="function"?o.setProperty("display","none","important"):o.display="none"):(f=oe.stateNode,g=oe.memoizedProps.style,i=g!=null&&g.hasOwnProperty("display")?g.display:null,f.style.display=w("display",i))}catch(Ce){nt(e,e.return,Ce)}}}else if(oe.tag===6){if(se===null)try{oe.stateNode.nodeValue=O?"":oe.memoizedProps}catch(Ce){nt(e,e.return,Ce)}}else if((oe.tag!==22&&oe.tag!==23||oe.memoizedState===null||oe===e)&&oe.child!==null){oe.child.return=oe,oe=oe.child;continue}if(oe===e)break e;for(;oe.sibling===null;){if(oe.return===null||oe.return===e)break e;se===oe&&(se=null),oe=oe.return}se===oe&&(se=null),oe.sibling.return=oe.return,oe=oe.sibling}}break;case 19:tn(n,e),cn(e),s&4&&ud(e);break;case 21:break;default:tn(n,e),cn(e)}}function cn(e){var n=e.flags;if(n&2){try{e:{for(var r=e.return;r!==null;){if(id(r)){var s=r;break e}r=r.return}throw Error(d(160))}switch(s.tag){case 5:var a=s.stateNode;s.flags&32&&($t(a,""),s.flags&=-33);var o=cd(e);dl(e,o,a);break;case 3:case 4:var i=s.stateNode.containerInfo,f=cd(e);cl(e,f,i);break;default:throw Error(d(161))}}catch(g){nt(e,e.return,g)}e.flags&=-3}n&4096&&(e.flags&=-4097)}function tf(e,n,r){we=e,fd(e)}function fd(e,n,r){for(var s=(e.mode&1)!==0;we!==null;){var a=we,o=a.child;if(a.tag===22&&s){var i=a.memoizedState!==null||ba;if(!i){var f=a.alternate,g=f!==null&&f.memoizedState!==null||wt;f=ba;var O=wt;if(ba=i,(wt=g)&&!O)for(we=a;we!==null;)i=we,g=i.child,i.tag===22&&i.memoizedState!==null?hd(a):g!==null?(g.return=i,we=g):hd(a);for(;o!==null;)we=o,fd(o),o=o.sibling;we=a,ba=f,wt=O}md(e)}else(a.subtreeFlags&8772)!==0&&o!==null?(o.return=a,we=o):md(e)}}function md(e){for(;we!==null;){var n=we;if((n.flags&8772)!==0){var r=n.alternate;try{if((n.flags&8772)!==0)switch(n.tag){case 0:case 11:case 15:wt||ja(5,n);break;case 1:var s=n.stateNode;if(n.flags&4&&!wt)if(r===null)s.componentDidMount();else{var a=n.elementType===n.type?r.memoizedProps:Zt(n.type,r.memoizedProps);s.componentDidUpdate(a,r.memoizedState,s.__reactInternalSnapshotBeforeUpdate)}var o=n.updateQueue;o!==null&&xc(n,o,s);break;case 3:var i=n.updateQueue;if(i!==null){if(r=null,n.child!==null)switch(n.child.tag){case 5:r=n.child.stateNode;break;case 1:r=n.child.stateNode}xc(n,i,r)}break;case 5:var f=n.stateNode;if(r===null&&n.flags&4){r=f;var g=n.memoizedProps;switch(n.type){case"button":case"input":case"select":case"textarea":g.autoFocus&&r.focus();break;case"img":g.src&&(r.src=g.src)}}break;case 6:break;case 4:break;case 12:break;case 13:if(n.memoizedState===null){var O=n.alternate;if(O!==null){var se=O.memoizedState;if(se!==null){var oe=se.dehydrated;oe!==null&&Zr(oe)}}}break;case 19:case 17:case 21:case 22:case 23:case 25:break;default:throw Error(d(163))}wt||n.flags&512&&il(n)}catch(ne){nt(n,n.return,ne)}}if(n===e){we=null;break}if(r=n.sibling,r!==null){r.return=n.return,we=r;break}we=n.return}}function xd(e){for(;we!==null;){var n=we;if(n===e){we=null;break}var r=n.sibling;if(r!==null){r.return=n.return,we=r;break}we=n.return}}function hd(e){for(;we!==null;){var n=we;try{switch(n.tag){case 0:case 11:case 15:var r=n.return;try{ja(4,n)}catch(g){nt(n,r,g)}break;case 1:var s=n.stateNode;if(typeof s.componentDidMount=="function"){var a=n.return;try{s.componentDidMount()}catch(g){nt(n,a,g)}}var o=n.return;try{il(n)}catch(g){nt(n,o,g)}break;case 5:var i=n.return;try{il(n)}catch(g){nt(n,i,g)}}}catch(g){nt(n,n.return,g)}if(n===e){we=null;break}var f=n.sibling;if(f!==null){f.return=n.return,we=f;break}we=n.return}}var nf=Math.ceil,wa=j.ReactCurrentDispatcher,ul=j.ReactCurrentOwner,Gt=j.ReactCurrentBatchConfig,$e=0,mt=null,ot=null,vt=0,At=0,Dr=Fn(0),ut=0,ws=null,ir=0,ka=0,pl=0,ks=null,Pt=null,fl=0,Or=1/0,wn=null,Sa=!1,ml=null,Vn=null,Na=!1,Bn=null,Ca=0,Ss=0,xl=null,_a=-1,za=0;function Ct(){return($e&6)!==0?rt():_a!==-1?_a:_a=rt()}function Wn(e){return(e.mode&1)===0?1:($e&2)!==0&&vt!==0?vt&-vt:Ap.transition!==null?(za===0&&(za=ci()),za):(e=Be,e!==0||(e=window.event,e=e===void 0?16:vi(e.type)),e)}function nn(e,n,r,s){if(50<Ss)throw Ss=0,xl=null,Error(d(185));Qr(e,r,s),(($e&2)===0||e!==mt)&&(e===mt&&(($e&2)===0&&(ka|=r),ut===4&&Gn(e,vt)),Tt(e,s),r===1&&$e===0&&(n.mode&1)===0&&(Or=rt()+500,na&&On()))}function Tt(e,n){var r=e.callbackNode;Au(e,n);var s=Ds(e,e===mt?vt:0);if(s===0)r!==null&&oi(r),e.callbackNode=null,e.callbackPriority=0;else if(n=s&-s,e.callbackPriority!==n){if(r!=null&&oi(r),n===1)e.tag===0?Op(vd.bind(null,e)):rc(vd.bind(null,e)),Mp(function(){($e&6)===0&&On()}),r=null;else{switch(di(s)){case 1:r=qa;break;case 4:r=li;break;case 16:r=Rs;break;case 536870912:r=ii;break;default:r=Rs}r=Cd(r,gd.bind(null,e))}e.callbackPriority=n,e.callbackNode=r}}function gd(e,n){if(_a=-1,za=0,($e&6)!==0)throw Error(d(327));var r=e.callbackNode;if(Ar()&&e.callbackNode!==r)return null;var s=Ds(e,e===mt?vt:0);if(s===0)return null;if((s&30)!==0||(s&e.expiredLanes)!==0||n)n=Ea(e,s);else{n=s;var a=$e;$e|=2;var o=bd();(mt!==e||vt!==n)&&(wn=null,Or=rt()+500,dr(e,n));do try{af();break}catch(f){yd(e,f)}while(!0);Ro(),wa.current=o,$e=a,ot!==null?n=0:(mt=null,vt=0,n=ut)}if(n!==0){if(n===2&&(a=Qa(e),a!==0&&(s=a,n=hl(e,a))),n===1)throw r=ws,dr(e,0),Gn(e,s),Tt(e,rt()),r;if(n===6)Gn(e,s);else{if(a=e.current.alternate,(s&30)===0&&!rf(a)&&(n=Ea(e,s),n===2&&(o=Qa(e),o!==0&&(s=o,n=hl(e,o))),n===1))throw r=ws,dr(e,0),Gn(e,s),Tt(e,rt()),r;switch(e.finishedWork=a,e.finishedLanes=s,n){case 0:case 1:throw Error(d(345));case 2:ur(e,Pt,wn);break;case 3:if(Gn(e,s),(s&130023424)===s&&(n=fl+500-rt(),10<n)){if(Ds(e,0)!==0)break;if(a=e.suspendedLanes,(a&s)!==s){Ct(),e.pingedLanes|=e.suspendedLanes&a;break}e.timeoutHandle=wo(ur.bind(null,e,Pt,wn),n);break}ur(e,Pt,wn);break;case 4:if(Gn(e,s),(s&4194240)===s)break;for(n=e.eventTimes,a=-1;0<s;){var i=31-Xt(s);o=1<<i,i=n[i],i>a&&(a=i),s&=~o}if(s=a,s=rt()-s,s=(120>s?120:480>s?480:1080>s?1080:1920>s?1920:3e3>s?3e3:4320>s?4320:1960*nf(s/1960))-s,10<s){e.timeoutHandle=wo(ur.bind(null,e,Pt,wn),s);break}ur(e,Pt,wn);break;case 5:ur(e,Pt,wn);break;default:throw Error(d(329))}}}return Tt(e,rt()),e.callbackNode===r?gd.bind(null,e):null}function hl(e,n){var r=ks;return e.current.memoizedState.isDehydrated&&(dr(e,n).flags|=256),e=Ea(e,n),e!==2&&(n=Pt,Pt=r,n!==null&&gl(n)),e}function gl(e){Pt===null?Pt=e:Pt.push.apply(Pt,e)}function rf(e){for(var n=e;;){if(n.flags&16384){var r=n.updateQueue;if(r!==null&&(r=r.stores,r!==null))for(var s=0;s<r.length;s++){var a=r[s],o=a.getSnapshot;a=a.value;try{if(!Kt(o(),a))return!1}catch{return!1}}}if(r=n.child,n.subtreeFlags&16384&&r!==null)r.return=n,n=r;else{if(n===e)break;for(;n.sibling===null;){if(n.return===null||n.return===e)return!0;n=n.return}n.sibling.return=n.return,n=n.sibling}}return!0}function Gn(e,n){for(n&=~pl,n&=~ka,e.suspendedLanes|=n,e.pingedLanes&=~n,e=e.expirationTimes;0<n;){var r=31-Xt(n),s=1<<r;e[r]=-1,n&=~s}}function vd(e){if(($e&6)!==0)throw Error(d(327));Ar();var n=Ds(e,0);if((n&1)===0)return Tt(e,rt()),null;var r=Ea(e,n);if(e.tag!==0&&r===2){var s=Qa(e);s!==0&&(n=s,r=hl(e,s))}if(r===1)throw r=ws,dr(e,0),Gn(e,n),Tt(e,rt()),r;if(r===6)throw Error(d(345));return e.finishedWork=e.current.alternate,e.finishedLanes=n,ur(e,Pt,wn),Tt(e,rt()),null}function vl(e,n){var r=$e;$e|=1;try{return e(n)}finally{$e=r,$e===0&&(Or=rt()+500,na&&On())}}function cr(e){Bn!==null&&Bn.tag===0&&($e&6)===0&&Ar();var n=$e;$e|=1;var r=Gt.transition,s=Be;try{if(Gt.transition=null,Be=1,e)return e()}finally{Be=s,Gt.transition=r,$e=n,($e&6)===0&&On()}}function yl(){At=Dr.current,Xe(Dr)}function dr(e,n){e.finishedWork=null,e.finishedLanes=0;var r=e.timeoutHandle;if(r!==-1&&(e.timeoutHandle=-1,Rp(r)),ot!==null)for(r=ot.return;r!==null;){var s=r;switch(zo(s),s.tag){case 1:s=s.type.childContextTypes,s!=null&&ea();break;case 3:Mr(),Xe(zt),Xe(yt),Uo();break;case 5:Ao(s);break;case 4:Mr();break;case 13:Xe(et);break;case 19:Xe(et);break;case 10:Mo(s.type._context);break;case 22:case 23:yl()}r=r.return}if(mt=e,ot=e=Hn(e.current,null),vt=At=n,ut=0,ws=null,pl=ka=ir=0,Pt=ks=null,ar!==null){for(n=0;n<ar.length;n++)if(r=ar[n],s=r.interleaved,s!==null){r.interleaved=null;var a=s.next,o=r.pending;if(o!==null){var i=o.next;o.next=a,s.next=i}r.pending=s}ar=null}return e}function yd(e,n){do{var r=ot;try{if(Ro(),pa.current=ha,fa){for(var s=tt.memoizedState;s!==null;){var a=s.queue;a!==null&&(a.pending=null),s=s.next}fa=!1}if(lr=0,ft=dt=tt=null,hs=!1,gs=0,ul.current=null,r===null||r.return===null){ut=1,ws=n,ot=null;break}e:{var o=e,i=r.return,f=r,g=n;if(n=vt,f.flags|=32768,g!==null&&typeof g=="object"&&typeof g.then=="function"){var O=g,se=f,oe=se.tag;if((se.mode&1)===0&&(oe===0||oe===11||oe===15)){var ne=se.alternate;ne?(se.updateQueue=ne.updateQueue,se.memoizedState=ne.memoizedState,se.lanes=ne.lanes):(se.updateQueue=null,se.memoizedState=null)}var je=Wc(i);if(je!==null){je.flags&=-257,Gc(je,i,f,o,n),je.mode&1&&Bc(o,O,n),n=je,g=O;var Se=n.updateQueue;if(Se===null){var Ce=new Set;Ce.add(g),n.updateQueue=Ce}else Se.add(g);break e}else{if((n&1)===0){Bc(o,O,n),bl();break e}g=Error(d(426))}}else if(Ze&&f.mode&1){var st=Wc(i);if(st!==null){(st.flags&65536)===0&&(st.flags|=256),Gc(st,i,f,o,n),Po(Lr(g,f));break e}}o=g=Lr(g,f),ut!==4&&(ut=2),ks===null?ks=[o]:ks.push(o),o=i;do{switch(o.tag){case 3:o.flags|=65536,n&=-n,o.lanes|=n;var z=Uc(o,g,n);mc(o,z);break e;case 1:f=g;var y=o.type,M=o.stateNode;if((o.flags&128)===0&&(typeof y.getDerivedStateFromError=="function"||M!==null&&typeof M.componentDidCatch=="function"&&(Vn===null||!Vn.has(M)))){o.flags|=65536,n&=-n,o.lanes|=n;var de=Vc(o,f,n);mc(o,de);break e}}o=o.return}while(o!==null)}wd(r)}catch(_e){n=_e,ot===r&&r!==null&&(ot=r=r.return);continue}break}while(!0)}function bd(){var e=wa.current;return wa.current=ha,e===null?ha:e}function bl(){(ut===0||ut===3||ut===2)&&(ut=4),mt===null||(ir&268435455)===0&&(ka&268435455)===0||Gn(mt,vt)}function Ea(e,n){var r=$e;$e|=2;var s=bd();(mt!==e||vt!==n)&&(wn=null,dr(e,n));do try{sf();break}catch(a){yd(e,a)}while(!0);if(Ro(),$e=r,wa.current=s,ot!==null)throw Error(d(261));return mt=null,vt=0,ut}function sf(){for(;ot!==null;)jd(ot)}function af(){for(;ot!==null&&!Iu();)jd(ot)}function jd(e){var n=Nd(e.alternate,e,At);e.memoizedProps=e.pendingProps,n===null?wd(e):ot=n,ul.current=null}function wd(e){var n=e;do{var r=n.alternate;if(e=n.return,(n.flags&32768)===0){if(r=Kp(r,n,At),r!==null){ot=r;return}}else{if(r=Jp(r,n),r!==null){r.flags&=32767,ot=r;return}if(e!==null)e.flags|=32768,e.subtreeFlags=0,e.deletions=null;else{ut=6,ot=null;return}}if(n=n.sibling,n!==null){ot=n;return}ot=n=e}while(n!==null);ut===0&&(ut=5)}function ur(e,n,r){var s=Be,a=Gt.transition;try{Gt.transition=null,Be=1,of(e,n,r,s)}finally{Gt.transition=a,Be=s}return null}function of(e,n,r,s){do Ar();while(Bn!==null);if(($e&6)!==0)throw Error(d(327));r=e.finishedWork;var a=e.finishedLanes;if(r===null)return null;if(e.finishedWork=null,e.finishedLanes=0,r===e.current)throw Error(d(177));e.callbackNode=null,e.callbackPriority=0;var o=r.lanes|r.childLanes;if($u(e,o),e===mt&&(ot=mt=null,vt=0),(r.subtreeFlags&2064)===0&&(r.flags&2064)===0||Na||(Na=!0,Cd(Rs,function(){return Ar(),null})),o=(r.flags&15990)!==0,(r.subtreeFlags&15990)!==0||o){o=Gt.transition,Gt.transition=null;var i=Be;Be=1;var f=$e;$e|=4,ul.current=null,ef(e,r),pd(r,e),Cp(bo),$s=!!yo,bo=yo=null,e.current=r,tf(r),Pu(),$e=f,Be=i,Gt.transition=o}else e.current=r;if(Na&&(Na=!1,Bn=e,Ca=a),o=e.pendingLanes,o===0&&(Vn=null),Mu(r.stateNode),Tt(e,rt()),n!==null)for(s=e.onRecoverableError,r=0;r<n.length;r++)a=n[r],s(a.value,{componentStack:a.stack,digest:a.digest});if(Sa)throw Sa=!1,e=ml,ml=null,e;return(Ca&1)!==0&&e.tag!==0&&Ar(),o=e.pendingLanes,(o&1)!==0?e===xl?Ss++:(Ss=0,xl=e):Ss=0,On(),null}function Ar(){if(Bn!==null){var e=di(Ca),n=Gt.transition,r=Be;try{if(Gt.transition=null,Be=16>e?16:e,Bn===null)var s=!1;else{if(e=Bn,Bn=null,Ca=0,($e&6)!==0)throw Error(d(331));var a=$e;for($e|=4,we=e.current;we!==null;){var o=we,i=o.child;if((we.flags&16)!==0){var f=o.deletions;if(f!==null){for(var g=0;g<f.length;g++){var O=f[g];for(we=O;we!==null;){var se=we;switch(se.tag){case 0:case 11:case 15:js(8,se,o)}var oe=se.child;if(oe!==null)oe.return=se,we=oe;else for(;we!==null;){se=we;var ne=se.sibling,je=se.return;if(ld(se),se===O){we=null;break}if(ne!==null){ne.return=je,we=ne;break}we=je}}}var Se=o.alternate;if(Se!==null){var Ce=Se.child;if(Ce!==null){Se.child=null;do{var st=Ce.sibling;Ce.sibling=null,Ce=st}while(Ce!==null)}}we=o}}if((o.subtreeFlags&2064)!==0&&i!==null)i.return=o,we=i;else e:for(;we!==null;){if(o=we,(o.flags&2048)!==0)switch(o.tag){case 0:case 11:case 15:js(9,o,o.return)}var z=o.sibling;if(z!==null){z.return=o.return,we=z;break e}we=o.return}}var y=e.current;for(we=y;we!==null;){i=we;var M=i.child;if((i.subtreeFlags&2064)!==0&&M!==null)M.return=i,we=M;else e:for(i=y;we!==null;){if(f=we,(f.flags&2048)!==0)try{switch(f.tag){case 0:case 11:case 15:ja(9,f)}}catch(_e){nt(f,f.return,_e)}if(f===i){we=null;break e}var de=f.sibling;if(de!==null){de.return=f.return,we=de;break e}we=f.return}}if($e=a,On(),sn&&typeof sn.onPostCommitFiberRoot=="function")try{sn.onPostCommitFiberRoot(Ms,e)}catch{}s=!0}return s}finally{Be=r,Gt.transition=n}}return!1}function kd(e,n,r){n=Lr(r,n),n=Uc(e,n,1),e=$n(e,n,1),n=Ct(),e!==null&&(Qr(e,1,n),Tt(e,n))}function nt(e,n,r){if(e.tag===3)kd(e,e,r);else for(;n!==null;){if(n.tag===3){kd(n,e,r);break}else if(n.tag===1){var s=n.stateNode;if(typeof n.type.getDerivedStateFromError=="function"||typeof s.componentDidCatch=="function"&&(Vn===null||!Vn.has(s))){e=Lr(r,e),e=Vc(n,e,1),n=$n(n,e,1),e=Ct(),n!==null&&(Qr(n,1,e),Tt(n,e));break}}n=n.return}}function lf(e,n,r){var s=e.pingCache;s!==null&&s.delete(n),n=Ct(),e.pingedLanes|=e.suspendedLanes&r,mt===e&&(vt&r)===r&&(ut===4||ut===3&&(vt&130023424)===vt&&500>rt()-fl?dr(e,0):pl|=r),Tt(e,n)}function Sd(e,n){n===0&&((e.mode&1)===0?n=1:(n=Fs,Fs<<=1,(Fs&130023424)===0&&(Fs=4194304)));var r=Ct();e=yn(e,n),e!==null&&(Qr(e,n,r),Tt(e,r))}function cf(e){var n=e.memoizedState,r=0;n!==null&&(r=n.retryLane),Sd(e,r)}function df(e,n){var r=0;switch(e.tag){case 13:var s=e.stateNode,a=e.memoizedState;a!==null&&(r=a.retryLane);break;case 19:s=e.stateNode;break;default:throw Error(d(314))}s!==null&&s.delete(n),Sd(e,r)}var Nd;Nd=function(e,n,r){if(e!==null)if(e.memoizedProps!==n.pendingProps||zt.current)It=!0;else{if((e.lanes&r)===0&&(n.flags&128)===0)return It=!1,Xp(e,n,r);It=(e.flags&131072)!==0}else It=!1,Ze&&(n.flags&1048576)!==0&&sc(n,sa,n.index);switch(n.lanes=0,n.tag){case 2:var s=n.type;ya(e,n),e=n.pendingProps;var a=_r(n,yt.current);Rr(n,r),a=Wo(null,n,s,e,a,r);var o=Go();return n.flags|=1,typeof a=="object"&&a!==null&&typeof a.render=="function"&&a.$$typeof===void 0?(n.tag=1,n.memoizedState=null,n.updateQueue=null,Et(s)?(o=!0,ta(n)):o=!1,n.memoizedState=a.state!==null&&a.state!==void 0?a.state:null,Do(n),a.updater=ga,n.stateNode=a,a._reactInternals=n,Ko(n,s,e,r),n=tl(null,n,s,!0,o,r)):(n.tag=0,Ze&&o&&_o(n),Nt(null,n,a,r),n=n.child),n;case 16:s=n.elementType;e:{switch(ya(e,n),e=n.pendingProps,a=s._init,s=a(s._payload),n.type=s,a=n.tag=pf(s),e=Zt(s,e),a){case 0:n=el(null,n,s,e,r);break e;case 1:n=Kc(null,n,s,e,r);break e;case 11:n=Hc(null,n,s,e,r);break e;case 14:n=qc(null,n,s,Zt(s.type,e),r);break e}throw Error(d(306,s,""))}return n;case 0:return s=n.type,a=n.pendingProps,a=n.elementType===s?a:Zt(s,a),el(e,n,s,a,r);case 1:return s=n.type,a=n.pendingProps,a=n.elementType===s?a:Zt(s,a),Kc(e,n,s,a,r);case 3:e:{if(Jc(n),e===null)throw Error(d(387));s=n.pendingProps,o=n.memoizedState,a=o.element,fc(e,n),da(n,s,null,r);var i=n.memoizedState;if(s=i.element,o.isDehydrated)if(o={element:s,isDehydrated:!1,cache:i.cache,pendingSuspenseBoundaries:i.pendingSuspenseBoundaries,transitions:i.transitions},n.updateQueue.baseState=o,n.memoizedState=o,n.flags&256){a=Lr(Error(d(423)),n),n=Zc(e,n,s,r,a);break e}else if(s!==a){a=Lr(Error(d(424)),n),n=Zc(e,n,s,r,a);break e}else for(Ot=Ln(n.stateNode.containerInfo.firstChild),Dt=n,Ze=!0,Jt=null,r=uc(n,null,s,r),n.child=r;r;)r.flags=r.flags&-3|4096,r=r.sibling;else{if(Ir(),s===a){n=jn(e,n,r);break e}Nt(e,n,s,r)}n=n.child}return n;case 5:return hc(n),e===null&&Io(n),s=n.type,a=n.pendingProps,o=e!==null?e.memoizedProps:null,i=a.children,jo(s,a)?i=null:o!==null&&jo(s,o)&&(n.flags|=32),Xc(e,n),Nt(e,n,i,r),n.child;case 6:return e===null&&Io(n),null;case 13:return ed(e,n,r);case 4:return Oo(n,n.stateNode.containerInfo),s=n.pendingProps,e===null?n.child=Pr(n,null,s,r):Nt(e,n,s,r),n.child;case 11:return s=n.type,a=n.pendingProps,a=n.elementType===s?a:Zt(s,a),Hc(e,n,s,a,r);case 7:return Nt(e,n,n.pendingProps,r),n.child;case 8:return Nt(e,n,n.pendingProps.children,r),n.child;case 12:return Nt(e,n,n.pendingProps.children,r),n.child;case 10:e:{if(s=n.type._context,a=n.pendingProps,o=n.memoizedProps,i=a.value,qe(la,s._currentValue),s._currentValue=i,o!==null)if(Kt(o.value,i)){if(o.children===a.children&&!zt.current){n=jn(e,n,r);break e}}else for(o=n.child,o!==null&&(o.return=n);o!==null;){var f=o.dependencies;if(f!==null){i=o.child;for(var g=f.firstContext;g!==null;){if(g.context===s){if(o.tag===1){g=bn(-1,r&-r),g.tag=2;var O=o.updateQueue;if(O!==null){O=O.shared;var se=O.pending;se===null?g.next=g:(g.next=se.next,se.next=g),O.pending=g}}o.lanes|=r,g=o.alternate,g!==null&&(g.lanes|=r),Lo(o.return,r,n),f.lanes|=r;break}g=g.next}}else if(o.tag===10)i=o.type===n.type?null:o.child;else if(o.tag===18){if(i=o.return,i===null)throw Error(d(341));i.lanes|=r,f=i.alternate,f!==null&&(f.lanes|=r),Lo(i,r,n),i=o.sibling}else i=o.child;if(i!==null)i.return=o;else for(i=o;i!==null;){if(i===n){i=null;break}if(o=i.sibling,o!==null){o.return=i.return,i=o;break}i=i.return}o=i}Nt(e,n,a.children,r),n=n.child}return n;case 9:return a=n.type,s=n.pendingProps.children,Rr(n,r),a=Bt(a),s=s(a),n.flags|=1,Nt(e,n,s,r),n.child;case 14:return s=n.type,a=Zt(s,n.pendingProps),a=Zt(s.type,a),qc(e,n,s,a,r);case 15:return Qc(e,n,n.type,n.pendingProps,r);case 17:return s=n.type,a=n.pendingProps,a=n.elementType===s?a:Zt(s,a),ya(e,n),n.tag=1,Et(s)?(e=!0,ta(n)):e=!1,Rr(n,r),Ac(n,s,a),Ko(n,s,a,r),tl(null,n,s,!0,e,r);case 19:return nd(e,n,r);case 22:return Yc(e,n,r)}throw Error(d(156,n.tag))};function Cd(e,n){return ai(e,n)}function uf(e,n,r,s){this.tag=e,this.key=r,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=n,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=s,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function Ht(e,n,r,s){return new uf(e,n,r,s)}function jl(e){return e=e.prototype,!(!e||!e.isReactComponent)}function pf(e){if(typeof e=="function")return jl(e)?1:0;if(e!=null){if(e=e.$$typeof,e===ge)return 11;if(e===fe)return 14}return 2}function Hn(e,n){var r=e.alternate;return r===null?(r=Ht(e.tag,n,e.key,e.mode),r.elementType=e.elementType,r.type=e.type,r.stateNode=e.stateNode,r.alternate=e,e.alternate=r):(r.pendingProps=n,r.type=e.type,r.flags=0,r.subtreeFlags=0,r.deletions=null),r.flags=e.flags&14680064,r.childLanes=e.childLanes,r.lanes=e.lanes,r.child=e.child,r.memoizedProps=e.memoizedProps,r.memoizedState=e.memoizedState,r.updateQueue=e.updateQueue,n=e.dependencies,r.dependencies=n===null?null:{lanes:n.lanes,firstContext:n.firstContext},r.sibling=e.sibling,r.index=e.index,r.ref=e.ref,r}function Ia(e,n,r,s,a,o){var i=2;if(s=e,typeof e=="function")jl(e)&&(i=1);else if(typeof e=="string")i=5;else e:switch(e){case h:return pr(r.children,a,o,n);case v:i=8,a|=8;break;case te:return e=Ht(12,r,n,a|2),e.elementType=te,e.lanes=o,e;case E:return e=Ht(13,r,n,a),e.elementType=E,e.lanes=o,e;case ue:return e=Ht(19,r,n,a),e.elementType=ue,e.lanes=o,e;case W:return Pa(r,a,o,n);default:if(typeof e=="object"&&e!==null)switch(e.$$typeof){case re:i=10;break e;case xe:i=9;break e;case ge:i=11;break e;case fe:i=14;break e;case ie:i=16,s=null;break e}throw Error(d(130,e==null?e:typeof e,""))}return n=Ht(i,r,n,a),n.elementType=e,n.type=s,n.lanes=o,n}function pr(e,n,r,s){return e=Ht(7,e,s,n),e.lanes=r,e}function Pa(e,n,r,s){return e=Ht(22,e,s,n),e.elementType=W,e.lanes=r,e.stateNode={isHidden:!1},e}function wl(e,n,r){return e=Ht(6,e,null,n),e.lanes=r,e}function kl(e,n,r){return n=Ht(4,e.children!==null?e.children:[],e.key,n),n.lanes=r,n.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},n}function ff(e,n,r,s,a){this.tag=n,this.containerInfo=e,this.finishedWork=this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.pendingContext=this.context=null,this.callbackPriority=0,this.eventTimes=Ya(0),this.expirationTimes=Ya(-1),this.entangledLanes=this.finishedLanes=this.mutableReadLanes=this.expiredLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=Ya(0),this.identifierPrefix=s,this.onRecoverableError=a,this.mutableSourceEagerHydrationData=null}function Sl(e,n,r,s,a,o,i,f,g){return e=new ff(e,n,r,f,g),n===1?(n=1,o===!0&&(n|=8)):n=0,o=Ht(3,null,null,n),e.current=o,o.stateNode=e,o.memoizedState={element:s,isDehydrated:r,cache:null,transitions:null,pendingSuspenseBoundaries:null},Do(o),e}function mf(e,n,r){var s=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:B,key:s==null?null:""+s,children:e,containerInfo:n,implementation:r}}function _d(e){if(!e)return Dn;e=e._reactInternals;e:{if(er(e)!==e||e.tag!==1)throw Error(d(170));var n=e;do{switch(n.tag){case 3:n=n.stateNode.context;break e;case 1:if(Et(n.type)){n=n.stateNode.__reactInternalMemoizedMergedChildContext;break e}}n=n.return}while(n!==null);throw Error(d(171))}if(e.tag===1){var r=e.type;if(Et(r))return tc(e,r,n)}return n}function zd(e,n,r,s,a,o,i,f,g){return e=Sl(r,s,!0,e,a,o,i,f,g),e.context=_d(null),r=e.current,s=Ct(),a=Wn(r),o=bn(s,a),o.callback=n??null,$n(r,o,a),e.current.lanes=a,Qr(e,a,s),Tt(e,s),e}function Ta(e,n,r,s){var a=n.current,o=Ct(),i=Wn(a);return r=_d(r),n.context===null?n.context=r:n.pendingContext=r,n=bn(o,i),n.payload={element:e},s=s===void 0?null:s,s!==null&&(n.callback=s),e=$n(a,n,i),e!==null&&(nn(e,a,i,o),ca(e,a,i)),i}function Ra(e){return e=e.current,e.child?(e.child.tag===5,e.child.stateNode):null}function Ed(e,n){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var r=e.retryLane;e.retryLane=r!==0&&r<n?r:n}}function Nl(e,n){Ed(e,n),(e=e.alternate)&&Ed(e,n)}function xf(){return null}var Id=typeof reportError=="function"?reportError:function(e){console.error(e)};function Cl(e){this._internalRoot=e}Ma.prototype.render=Cl.prototype.render=function(e){var n=this._internalRoot;if(n===null)throw Error(d(409));Ta(e,n,null,null)},Ma.prototype.unmount=Cl.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var n=e.containerInfo;cr(function(){Ta(null,e,null,null)}),n[xn]=null}};function Ma(e){this._internalRoot=e}Ma.prototype.unstable_scheduleHydration=function(e){if(e){var n=fi();e={blockedOn:null,target:e,priority:n};for(var r=0;r<Tn.length&&n!==0&&n<Tn[r].priority;r++);Tn.splice(r,0,e),r===0&&hi(e)}};function _l(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function La(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11&&(e.nodeType!==8||e.nodeValue!==" react-mount-point-unstable "))}function Pd(){}function hf(e,n,r,s,a){if(a){if(typeof s=="function"){var o=s;s=function(){var O=Ra(i);o.call(O)}}var i=zd(n,s,e,0,null,!1,!1,"",Pd);return e._reactRootContainer=i,e[xn]=i.current,is(e.nodeType===8?e.parentNode:e),cr(),i}for(;a=e.lastChild;)e.removeChild(a);if(typeof s=="function"){var f=s;s=function(){var O=Ra(g);f.call(O)}}var g=Sl(e,0,!1,null,null,!1,!1,"",Pd);return e._reactRootContainer=g,e[xn]=g.current,is(e.nodeType===8?e.parentNode:e),cr(function(){Ta(n,g,r,s)}),g}function Fa(e,n,r,s,a){var o=r._reactRootContainer;if(o){var i=o;if(typeof a=="function"){var f=a;a=function(){var g=Ra(i);f.call(g)}}Ta(n,i,e,a)}else i=hf(r,n,e,a,s);return Ra(i)}ui=function(e){switch(e.tag){case 3:var n=e.stateNode;if(n.current.memoizedState.isDehydrated){var r=qr(n.pendingLanes);r!==0&&(Xa(n,r|1),Tt(n,rt()),($e&6)===0&&(Or=rt()+500,On()))}break;case 13:cr(function(){var s=yn(e,1);if(s!==null){var a=Ct();nn(s,e,1,a)}}),Nl(e,1)}},Ka=function(e){if(e.tag===13){var n=yn(e,134217728);if(n!==null){var r=Ct();nn(n,e,134217728,r)}Nl(e,134217728)}},pi=function(e){if(e.tag===13){var n=Wn(e),r=yn(e,n);if(r!==null){var s=Ct();nn(r,e,n,s)}Nl(e,n)}},fi=function(){return Be},mi=function(e,n){var r=Be;try{return Be=e,n()}finally{Be=r}},St=function(e,n,r){switch(n){case"input":if(Je(e,r),n=r.name,r.type==="radio"&&n!=null){for(r=e;r.parentNode;)r=r.parentNode;for(r=r.querySelectorAll("input[name="+JSON.stringify(""+n)+'][type="radio"]'),n=0;n<r.length;n++){var s=r[n];if(s!==e&&s.form===e.form){var a=Zs(s);if(!a)throw Error(d(90));pe(s),Je(s,a)}}}break;case"textarea":hr(e,r);break;case"select":n=r.value,n!=null&&kt(e,!!r.multiple,n,!1)}},Is=vl,Ps=cr;var gf={usingClientEntryPoint:!1,Events:[us,Nr,Zs,Gr,Hr,vl]},Ns={findFiberByHostInstance:tr,bundleType:0,version:"18.3.1",rendererPackageName:"react-dom"},vf={bundleType:Ns.bundleType,version:Ns.version,rendererPackageName:Ns.rendererPackageName,rendererConfig:Ns.rendererConfig,overrideHookState:null,overrideHookStateDeletePath:null,overrideHookStateRenamePath:null,overrideProps:null,overridePropsDeletePath:null,overridePropsRenamePath:null,setErrorHandler:null,setSuspenseHandler:null,scheduleUpdate:null,currentDispatcherRef:j.ReactCurrentDispatcher,findHostInstanceByFiber:function(e){return e=ri(e),e===null?null:e.stateNode},findFiberByHostInstance:Ns.findFiberByHostInstance||xf,findHostInstancesForRefresh:null,scheduleRefresh:null,scheduleRoot:null,setRefreshHandler:null,getCurrentFiber:null,reconcilerVersion:"18.3.1-next-f1338f8080-20240426"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"){var Da=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Da.isDisabled&&Da.supportsFiber)try{Ms=Da.inject(vf),sn=Da}catch{}}return Rt.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=gf,Rt.createPortal=function(e,n){var r=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!_l(n))throw Error(d(200));return mf(e,n,null,r)},Rt.createRoot=function(e,n){if(!_l(e))throw Error(d(299));var r=!1,s="",a=Id;return n!=null&&(n.unstable_strictMode===!0&&(r=!0),n.identifierPrefix!==void 0&&(s=n.identifierPrefix),n.onRecoverableError!==void 0&&(a=n.onRecoverableError)),n=Sl(e,1,!1,null,null,r,!1,s,a),e[xn]=n.current,is(e.nodeType===8?e.parentNode:e),new Cl(n)},Rt.findDOMNode=function(e){if(e==null)return null;if(e.nodeType===1)return e;var n=e._reactInternals;if(n===void 0)throw typeof e.render=="function"?Error(d(188)):(e=Object.keys(e).join(","),Error(d(268,e)));return e=ri(n),e=e===null?null:e.stateNode,e},Rt.flushSync=function(e){return cr(e)},Rt.hydrate=function(e,n,r){if(!La(n))throw Error(d(200));return Fa(null,e,n,!0,r)},Rt.hydrateRoot=function(e,n,r){if(!_l(e))throw Error(d(405));var s=r!=null&&r.hydratedSources||null,a=!1,o="",i=Id;if(r!=null&&(r.unstable_strictMode===!0&&(a=!0),r.identifierPrefix!==void 0&&(o=r.identifierPrefix),r.onRecoverableError!==void 0&&(i=r.onRecoverableError)),n=zd(n,null,e,1,r??null,a,!1,o,i),e[xn]=n.current,is(e),s)for(e=0;e<s.length;e++)r=s[e],a=r._getVersion,a=a(r._source),n.mutableSourceEagerHydrationData==null?n.mutableSourceEagerHydrationData=[r,a]:n.mutableSourceEagerHydrationData.push(r,a);return new Ma(n)},Rt.render=function(e,n,r){if(!La(n))throw Error(d(200));return Fa(null,e,n,!1,r)},Rt.unmountComponentAtNode=function(e){if(!La(e))throw Error(d(40));return e._reactRootContainer?(cr(function(){Fa(null,null,e,!1,function(){e._reactRootContainer=null,e[xn]=null})}),!0):!1},Rt.unstable_batchedUpdates=vl,Rt.unstable_renderSubtreeIntoContainer=function(e,n,r,s){if(!La(r))throw Error(d(200));if(e==null||e._reactInternals===void 0)throw Error(d(38));return Fa(e,n,r,!1,s)},Rt.version="18.3.1-next-f1338f8080-20240426",Rt}var Ad;function _f(){if(Ad)return Il.exports;Ad=1;function c(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(c)}catch(x){console.error(x)}}return c(),Il.exports=Cf(),Il.exports}var $d;function zf(){if($d)return Oa;$d=1;var c=_f();return Oa.createRoot=c.createRoot,Oa.hydrateRoot=c.hydrateRoot,Oa}var Ef=zf();const If=iu(Ef);const Pf=c=>c.replace(/([a-z0-9])([A-Z])/g,"$1-$2").toLowerCase(),cu=(...c)=>c.filter((x,d,N)=>!!x&&x.trim()!==""&&N.indexOf(x)===d).join(" ").trim();var Tf={xmlns:"http://www.w3.org/2000/svg",width:24,height:24,viewBox:"0 0 24 24",fill:"none",stroke:"currentColor",strokeWidth:2,strokeLinecap:"round",strokeLinejoin:"round"};const Rf=l.forwardRef(({color:c="currentColor",size:x=24,strokeWidth:d=2,absoluteStrokeWidth:N,className:p="",children:b,iconNode:S,...R},I)=>l.createElement("svg",{ref:I,...Tf,width:x,height:x,stroke:c,strokeWidth:N?Number(d)*24/Number(x):d,className:cu("lucide",p),...R},[...S.map(([P,T])=>l.createElement(P,T)),...Array.isArray(b)?b:[b]]));const ye=(c,x)=>{const d=l.forwardRef(({className:N,...p},b)=>l.createElement(Rf,{ref:b,iconNode:x,className:cu(`lucide-${Pf(c)}`,N),...p}));return d.displayName=`${c}`,d};const Mf=[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"m12 5 7 7-7 7",key:"xquz4c"}]],Lf=ye("ArrowRight",Mf);const Ff=[["path",{d:"m21 16-4 4-4-4",key:"f6ql7i"}],["path",{d:"M17 20V4",key:"1ejh1v"}],["path",{d:"m3 8 4-4 4 4",key:"11wl7u"}],["path",{d:"M7 4v16",key:"1glfcx"}]],Df=ye("ArrowUpDown",Ff);const Of=[["path",{d:"M20 6 9 17l-5-5",key:"1gmf2c"}]],mr=ye("Check",Of);const Af=[["path",{d:"m6 9 6 6 6-6",key:"qrunsl"}]],Qt=ye("ChevronDown",Af);const $f=[["path",{d:"m15 18-6-6 6-6",key:"1wnfg3"}]],Uf=ye("ChevronLeft",$f);const Vf=[["path",{d:"m9 18 6-6-6-6",key:"mthhwq"}]],Bf=ye("ChevronRight",Vf);const Wf=[["path",{d:"m18 15-6-6-6 6",key:"153udz"}]],Gf=ye("ChevronUp",Wf);const Hf=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["line",{x1:"12",x2:"12",y1:"8",y2:"12",key:"1pkeuh"}],["line",{x1:"12",x2:"12.01",y1:"16",y2:"16",key:"4dfq90"}]],Yl=ye("CircleAlert",Hf);const qf=[["path",{d:"M21.801 10A10 10 0 1 1 17 3.335",key:"yps3ct"}],["path",{d:"m9 11 3 3L22 4",key:"1pflzl"}]],du=ye("CircleCheckBig",qf);const Qf=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"m9 12 2 2 4-4",key:"dzmm74"}]],Yf=ye("CircleCheck",Qf);const Xf=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3",key:"1u773s"}],["path",{d:"M12 17h.01",key:"p32p05"}]],Kf=ye("CircleHelp",Xf);const Jf=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"m15 9-6 6",key:"1uzhvr"}],["path",{d:"m9 9 6 6",key:"z0biqf"}]],Zf=ye("CircleX",Jf);const em=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["polyline",{points:"12 6 12 12 16 14",key:"68esgv"}]],Qn=ye("Clock",em);const tm=[["circle",{cx:"8",cy:"8",r:"6",key:"3yglwk"}],["path",{d:"M18.09 10.37A6 6 0 1 1 10.34 18",key:"t5s6rm"}],["path",{d:"M7 6h1v4",key:"1obek4"}],["path",{d:"m16.71 13.88.7.71-2.82 2.82",key:"1rbuyh"}]],$l=ye("Coins",tm);const nm=[["rect",{width:"14",height:"14",x:"8",y:"8",rx:"2",ry:"2",key:"17jyea"}],["path",{d:"M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2",key:"zix9uf"}]],un=ye("Copy",nm);const rm=[["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",key:"ih7n3h"}],["polyline",{points:"7 10 12 15 17 10",key:"2ggqvy"}],["line",{x1:"12",x2:"12",y1:"15",y2:"3",key:"1vk2je"}]],qt=ye("Download",rm);const sm=[["path",{d:"M15 3h6v6",key:"1q9fwt"}],["path",{d:"M10 14 21 3",key:"gplh6r"}],["path",{d:"M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6",key:"a6xqqp"}]],Ul=ye("ExternalLink",sm);const am=[["path",{d:"M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0",key:"1nclc0"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]],uu=ye("Eye",am);const om=[["path",{d:"M17.5 22h.5a2 2 0 0 0 2-2V7l-5-5H6a2 2 0 0 0-2 2v3",key:"rslqgf"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"M2 19a2 2 0 1 1 4 0v1a2 2 0 1 1-4 0v-4a6 6 0 0 1 12 0v4a2 2 0 1 1-4 0v-1a2 2 0 1 1 4 0",key:"9f7x3i"}]],Vl=ye("FileAudio",om);const lm=[["path",{d:"M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",key:"1rqfz7"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"M10 12a1 1 0 0 0-1 1v1a1 1 0 0 1-1 1 1 1 0 0 1 1 1v1a1 1 0 0 0 1 1",key:"1oajmo"}],["path",{d:"M14 18a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1 1 1 0 0 1-1-1v-1a1 1 0 0 0-1-1",key:"mpwhp6"}]],Ud=ye("FileJson",lm);const im=[["path",{d:"M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",key:"1rqfz7"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"M10 9H8",key:"b1mrlr"}],["path",{d:"M16 13H8",key:"t4e002"}],["path",{d:"M16 17H8",key:"z1uh3a"}]],Bl=ye("FileText",im);const cm=[["path",{d:"M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z",key:"1rqfz7"}],["path",{d:"M14 2v4a2 2 0 0 0 2 2h4",key:"tnqrlb"}],["path",{d:"m10 11 5 3-5 3v-6Z",key:"7ntvm4"}]],dm=ye("FileVideo",cm);const um=[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",key:"afitv7"}],["path",{d:"M7 3v18",key:"bbkbws"}],["path",{d:"M3 7.5h4",key:"zfgn84"}],["path",{d:"M3 12h18",key:"1i2n21"}],["path",{d:"M3 16.5h4",key:"1230mu"}],["path",{d:"M17 3v18",key:"in4fa5"}],["path",{d:"M17 7.5h4",key:"myr1c1"}],["path",{d:"M17 16.5h4",key:"go4c1d"}]],Ua=ye("Film",um);const pm=[["polygon",{points:"22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3",key:"1yg77f"}]],pu=ye("Filter",pm);const fm=[["path",{d:"m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2",key:"usdka0"}]],mm=ye("FolderOpen",fm);const xm=[["line",{x1:"22",x2:"2",y1:"6",y2:"6",key:"15w7dq"}],["line",{x1:"22",x2:"2",y1:"18",y2:"18",key:"1ip48p"}],["line",{x1:"6",x2:"6",y1:"2",y2:"22",key:"a2lnyx"}],["line",{x1:"18",x2:"18",y1:"2",y2:"22",key:"8vb6jd"}]],hm=ye("Frame",xm);const gm=[["path",{d:"M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z",key:"c3ymky"}]],Ur=ye("Heart",gm);const vm=[["path",{d:"M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8",key:"1357e3"}],["path",{d:"M3 3v5h5",key:"1xhq8a"}],["path",{d:"M12 7v5l4 2",key:"1fdv2h"}]],ym=ye("History",vm);const bm=[["rect",{width:"18",height:"18",x:"3",y:"3",rx:"2",ry:"2",key:"1m3agn"}],["circle",{cx:"9",cy:"9",r:"2",key:"af1f0g"}],["path",{d:"m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21",key:"1xmnt7"}]],Nn=ye("Image",bm);const jm=[["path",{d:"M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z",key:"zw3jo"}],["path",{d:"M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12",key:"1wduqc"}],["path",{d:"M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17",key:"kqbvx6"}]],fu=ye("Layers",jm);const wm=[["path",{d:"M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71",key:"1cjeqo"}],["path",{d:"M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",key:"19qd67"}]],mu=ye("Link",wm);const km=[["path",{d:"M21 12a9 9 0 1 1-6.219-8.56",key:"13zald"}]],at=ye("LoaderCircle",km);const Sm=[["path",{d:"M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4",key:"u53s6r"}],["polyline",{points:"10 17 15 12 10 7",key:"1ail0h"}],["line",{x1:"15",x2:"3",y1:"12",y2:"12",key:"v6grx8"}]],Nm=ye("LogIn",Sm);const Cm=[["path",{d:"M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4",key:"1uf3rs"}],["polyline",{points:"16 17 21 12 16 7",key:"1gabdz"}],["line",{x1:"21",x2:"9",y1:"12",y2:"12",key:"1uyos4"}]],_m=ye("LogOut",Cm);const zm=[["polyline",{points:"15 3 21 3 21 9",key:"mznyad"}],["polyline",{points:"9 21 3 21 3 15",key:"1avn1i"}],["line",{x1:"21",x2:"14",y1:"3",y2:"10",key:"ota7mn"}],["line",{x1:"3",x2:"10",y1:"21",y2:"14",key:"1atl0r"}]],Em=ye("Maximize2",zm);const Im=[["path",{d:"M7.9 20A9 9 0 1 0 4 16.1L2 22Z",key:"vv11sd"}]],Pm=ye("MessageCircle",Im);const Tm=[["path",{d:"M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",key:"1lielz"}]],Rl=ye("MessageSquare",Tm);const Rm=[["path",{d:"M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z",key:"131961"}],["path",{d:"M19 10v2a7 7 0 0 1-14 0v-2",key:"1vc78b"}],["line",{x1:"12",x2:"12",y1:"19",y2:"22",key:"x3vr5v"}]],Xl=ye("Mic",Rm);const Mm=[["polyline",{points:"4 14 10 14 10 20",key:"11kfnr"}],["polyline",{points:"20 10 14 10 14 4",key:"rlmsce"}],["line",{x1:"14",x2:"21",y1:"10",y2:"3",key:"o5lafz"}],["line",{x1:"3",x2:"10",y1:"21",y2:"14",key:"1atl0r"}]],Lm=ye("Minimize2",Mm);const Fm=[["path",{d:"M12 2v20",key:"t6zp3m"}],["path",{d:"m15 19-3 3-3-3",key:"11eu04"}],["path",{d:"m19 9 3 3-3 3",key:"1mg7y2"}],["path",{d:"M2 12h20",key:"9i4pu4"}],["path",{d:"m5 9-3 3 3 3",key:"j64kie"}],["path",{d:"m9 5 3-3 3 3",key:"l8vdw6"}]],Dm=ye("Move",Fm);const Om=[["path",{d:"M9 18V5l12-2v13",key:"1jmyc2"}],["circle",{cx:"6",cy:"18",r:"3",key:"fqmcym"}],["circle",{cx:"18",cy:"16",r:"3",key:"1hluhg"}]],Am=ye("Music",Om);const $m=[["rect",{x:"14",y:"4",width:"4",height:"16",rx:"1",key:"zuxfzm"}],["rect",{x:"6",y:"4",width:"4",height:"16",rx:"1",key:"1okwgv"}]],Wl=ye("Pause",$m);const Um=[["polygon",{points:"6 3 20 12 6 21 6 3",key:"1oa8hb"}]],Va=ye("Play",Um);const Vm=[["path",{d:"M5 12h14",key:"1ays0h"}],["path",{d:"M12 5v14",key:"s699le"}]],Vd=ye("Plus",Vm);const Bm=[["path",{d:"M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8",key:"v9h5vc"}],["path",{d:"M21 3v5h-5",key:"1q7to0"}],["path",{d:"M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16",key:"3uifl3"}],["path",{d:"M8 16H3v5",key:"1cv678"}]],_s=ye("RefreshCw",Bm);const Wm=[["path",{d:"m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z",key:"7g6ntu"}],["path",{d:"m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z",key:"ijws7r"}],["path",{d:"M7 21h10",key:"1b0cd5"}],["path",{d:"M12 3v18",key:"108xh3"}],["path",{d:"M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2",key:"3gwbw2"}]],Gm=ye("Scale",Wm);const Hm=[["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}],["path",{d:"m21 21-4.3-4.3",key:"1qie3q"}]],qm=ye("Search",Hm);const Qm=[["path",{d:"M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z",key:"1ffxy3"}],["path",{d:"m21.854 2.147-10.94 10.939",key:"12cjpa"}]],xu=ye("Send",Qm);const Ym=[["path",{d:"M20 7h-9",key:"3s1dr2"}],["path",{d:"M14 17H5",key:"gfn3mx"}],["circle",{cx:"17",cy:"17",r:"3",key:"18b49y"}],["circle",{cx:"7",cy:"7",r:"3",key:"dfmy0x"}]],hu=ye("Settings2",Ym);const Xm=[["path",{d:"M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z",key:"1qme2f"}],["circle",{cx:"12",cy:"12",r:"3",key:"1v7zrd"}]],Yn=ye("Settings",Xm);const Km=[["circle",{cx:"18",cy:"5",r:"3",key:"gq8acd"}],["circle",{cx:"6",cy:"12",r:"3",key:"w7nqdw"}],["circle",{cx:"18",cy:"19",r:"3",key:"1xt0gg"}],["line",{x1:"8.59",x2:"15.42",y1:"13.51",y2:"17.49",key:"47mynk"}],["line",{x1:"15.41",x2:"8.59",y1:"6.51",y2:"10.49",key:"1n3mei"}]],Jm=ye("Share2",Km);const Zm=[["path",{d:"M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z",key:"oel41y"}]],ex=ye("Shield",Zm);const tx=[["line",{x1:"4",x2:"4",y1:"21",y2:"14",key:"1p332r"}],["line",{x1:"4",x2:"4",y1:"10",y2:"3",key:"gb41h5"}],["line",{x1:"12",x2:"12",y1:"21",y2:"12",key:"hf2csr"}],["line",{x1:"12",x2:"12",y1:"8",y2:"3",key:"1kfi7u"}],["line",{x1:"20",x2:"20",y1:"21",y2:"16",key:"1lhrwl"}],["line",{x1:"20",x2:"20",y1:"12",y2:"3",key:"16vvfq"}],["line",{x1:"2",x2:"6",y1:"14",y2:"14",key:"1uebub"}],["line",{x1:"10",x2:"14",y1:"8",y2:"8",key:"1yglbp"}],["line",{x1:"18",x2:"22",y1:"16",y2:"16",key:"1jxqpz"}]],zs=ye("SlidersVertical",tx);const nx=[["circle",{cx:"12",cy:"12",r:"10",key:"1mglay"}],["path",{d:"M8 14s1.5 2 4 2 4-2 4-2",key:"1y1vjs"}],["line",{x1:"9",x2:"9.01",y1:"9",y2:"9",key:"yxxnd0"}],["line",{x1:"15",x2:"15.01",y1:"9",y2:"9",key:"1p4y9e"}]],rx=ye("Smile",nx);const sx=[["path",{d:"M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z",key:"4pj2yx"}],["path",{d:"M20 3v4",key:"1olli1"}],["path",{d:"M22 5h-4",key:"1gvqau"}],["path",{d:"M4 17v2",key:"vumght"}],["path",{d:"M5 18H3",key:"zchphs"}]],Cn=ye("Sparkles",sx);const ax=[["path",{d:"M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 .586 1.414l8.704 8.704a2.426 2.426 0 0 0 3.42 0l6.58-6.58a2.426 2.426 0 0 0 0-3.42z",key:"vktsd0"}],["circle",{cx:"7.5",cy:"7.5",r:".5",fill:"currentColor",key:"kqv944"}]],ox=ye("Tag",ax);const lx=[["polyline",{points:"4 17 10 11 4 5",key:"akl6gq"}],["line",{x1:"12",x2:"20",y1:"19",y2:"19",key:"q2wloq"}]],Bd=ye("Terminal",lx);const ix=[["path",{d:"M3 6h18",key:"d0wm0j"}],["path",{d:"M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6",key:"4alrt4"}],["path",{d:"M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2",key:"v07s0e"}],["line",{x1:"10",x2:"10",y1:"11",y2:"17",key:"1uufr5"}],["line",{x1:"14",x2:"14",y1:"11",y2:"17",key:"xtxkd"}]],Ba=ye("Trash2",ix);const cx=[["polyline",{points:"22 7 13.5 15.5 8.5 10.5 2 17",key:"126l90"}],["polyline",{points:"16 7 22 7 22 13",key:"kwv8wd"}]],dx=ye("TrendingUp",cx);const ux=[["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",key:"ih7n3h"}],["polyline",{points:"17 8 12 3 7 8",key:"t8dd8p"}],["line",{x1:"12",x2:"12",y1:"3",y2:"15",key:"widbto"}]],pt=ye("Upload",ux);const px=[["path",{d:"M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2",key:"975kel"}],["circle",{cx:"12",cy:"7",r:"4",key:"17ys0d"}]],Gl=ye("User",px);const fx=[["path",{d:"m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5",key:"ftymec"}],["rect",{x:"2",y:"6",width:"14",height:"12",rx:"2",key:"158x01"}]],Xn=ye("Video",fx);const mx=[["path",{d:"M11 4.702a.705.705 0 0 0-1.203-.498L6.413 7.587A1.4 1.4 0 0 1 5.416 8H3a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2.416a1.4 1.4 0 0 1 .997.413l3.383 3.384A.705.705 0 0 0 11 19.298z",key:"uqj9uw"}],["path",{d:"M16 9a5 5 0 0 1 0 6",key:"1q6k2b"}],["path",{d:"M19.364 18.364a9 9 0 0 0 0-12.728",key:"ijwkga"}]],Br=ye("Volume2",mx);const xx=[["path",{d:"m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72",key:"ul74o6"}],["path",{d:"m14 7 3 3",key:"1r5n42"}],["path",{d:"M5 6v4",key:"ilb8ba"}],["path",{d:"M19 14v4",key:"blhpug"}],["path",{d:"M10 2v2",key:"7u0qdc"}],["path",{d:"M7 8H3",key:"zfb6yr"}],["path",{d:"M21 16h-4",key:"1cnmox"}],["path",{d:"M11 3H9",key:"1obp7u"}]],xr=ye("WandSparkles",xx);const hx=[["path",{d:"M12 20h.01",key:"zekei9"}],["path",{d:"M8.5 16.429a5 5 0 0 1 7 0",key:"1bycff"}],["path",{d:"M5 12.859a10 10 0 0 1 5.17-2.69",key:"1dl1wf"}],["path",{d:"M19 12.859a10 10 0 0 0-2.007-1.523",key:"4k23kn"}],["path",{d:"M2 8.82a15 15 0 0 1 4.177-2.643",key:"1grhjp"}],["path",{d:"M22 8.82a15 15 0 0 0-11.288-3.764",key:"z3jwby"}],["path",{d:"m2 2 20 20",key:"1ooewy"}]],gx=ye("WifiOff",hx);const vx=[["path",{d:"M12 20h.01",key:"zekei9"}],["path",{d:"M2 8.82a15 15 0 0 1 20 0",key:"dnpr2z"}],["path",{d:"M5 12.859a10 10 0 0 1 14 0",key:"1x1e6c"}],["path",{d:"M8.5 16.429a5 5 0 0 1 7 0",key:"1bycff"}]],yx=ye("Wifi",vx);const bx=[["rect",{width:"8",height:"8",x:"3",y:"3",rx:"2",key:"by2w9f"}],["path",{d:"M7 11v4a2 2 0 0 0 2 2h4",key:"xkn7yn"}],["rect",{width:"8",height:"8",x:"13",y:"13",rx:"2",key:"1cgmvn"}]],jx=ye("Workflow",bx);const wx=[["path",{d:"M18 6 6 18",key:"1bl5f8"}],["path",{d:"m6 6 12 12",key:"d8bk6v"}]],lt=ye("X",wx);const kx=[["path",{d:"M2.5 17a24.12 24.12 0 0 1 0-10 2 2 0 0 1 1.4-1.4 49.56 49.56 0 0 1 16.2 0A2 2 0 0 1 21.5 7a24.12 24.12 0 0 1 0 10 2 2 0 0 1-1.4 1.4 49.55 49.55 0 0 1-16.2 0A2 2 0 0 1 2.5 17",key:"1q2vi4"}],["path",{d:"m10 15 5-3-5-3z",key:"1jp15x"}]],Sx=ye("Youtube",kx);const Nx=[["path",{d:"M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z",key:"1xq2db"}]],Kl=ye("Zap",Nx);const Cx=[["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}],["line",{x1:"21",x2:"16.65",y1:"21",y2:"16.65",key:"13gj7c"}],["line",{x1:"11",x2:"11",y1:"8",y2:"14",key:"1vmskp"}],["line",{x1:"8",x2:"14",y1:"11",y2:"11",key:"durymu"}]],Hl=ye("ZoomIn",Cx),_x=window.location.hostname==="oelala.xyz",ve=_x?"https://api.oelala.xyz":"http://192.168.1.2:7998",Me=!1,me={TEXT_TO_IMAGE:"text-to-image",TEXT_TO_VIDEO:"text-to-video",IMAGE_TO_VIDEO:"image-to-video",TEXT_TO_IMAGE_TO_VIDEO:"text-to-image-to-video",VIDEO_TO_VIDEO:"video-to-video",SPEECH_TO_VIDEO:"speech-to-video",VIDEO_UPSCALER:"video-upscaler",FRAME_INTERPOLATION:"frame-interpolation",IMAGE_TO_IMAGE:"image-to-image",REFRAME:"reframe",FACE_SWAP:"face-swap",UPSCALER:"upscaler",PROMPT_GENERATOR:"prompt-generator",IMAGE_TO_TEXT:"image-to-text",VIDEO_TO_TEXT:"video-to-text",AUDIO_GENERATION:"audio-generation",VOICE_CLONING:"voice-cloning",LIP_SYNC:"lip-sync",PIPELINE:"pipeline",LORA_TRAINING:"lora-training",GALLERY:"gallery",MY_MEDIA_ALL:"my-media-all",MY_MEDIA_VIDEOS:"my-media-videos",MY_MEDIA_IMAGES:"my-media-images",MY_MEDIA_AUDIO:"my-media-audio",MY_MEDIA_PROMPTS:"my-media-prompts"},zx=[{id:"video-tools",title:"Video Tools",items:[{id:me.IMAGE_TO_VIDEO,label:"Image to Video",status:"ready"},{id:me.TEXT_TO_VIDEO,label:"Text to Video",status:"ready"},{id:me.TEXT_TO_IMAGE_TO_VIDEO,label:"Text to Image to Video",status:"ready"},{id:me.VIDEO_TO_VIDEO,label:"Video to Video",status:"ready"},{id:me.VIDEO_UPSCALER,label:"Video Upscaler",status:"new"},{id:me.FRAME_INTERPOLATION,label:"Frame Interpolation",status:"new"},{id:me.SPEECH_TO_VIDEO,label:"Speech to Video",status:"new"}]},{id:"image-tools",title:"Image Tools",items:[{id:me.TEXT_TO_IMAGE,label:"Text to Image",status:"ready"},{id:me.IMAGE_TO_IMAGE,label:"Image to Image",status:"ready"},{id:me.UPSCALER,label:"Upscaler",status:"ready"},{id:me.REFRAME,label:"Reframe",status:"new"},{id:me.FACE_SWAP,label:"Face Swap",status:"new"}]},{id:"prompt-tools",title:"Prompt Tools",items:[{id:me.PROMPT_GENERATOR,label:"Prompt Generator",status:"new"},{id:me.IMAGE_TO_TEXT,label:"Image to Text",status:"new"},{id:me.VIDEO_TO_TEXT,label:"Video to Text",status:"new"}]},{id:"audio-tools",title:"Audio Tools",items:[{id:me.AUDIO_GENERATION,label:"Audio Generation",status:"new"},{id:me.VOICE_CLONING,label:"Voice Cloning",status:"new"},{id:me.LIP_SYNC,label:"Lip Sync",status:"new"}]},{id:"advanced",title:"Advanced",items:[{id:me.PIPELINE,label:"Pipeline",status:"ready"},{id:me.LORA_TRAINING,label:"LoRA Training",status:"ready"}]},{id:"community",title:"Community",items:[{id:me.GALLERY,label:"Gallery",status:"new",emoji:"🖼️"}]},{id:"my-media",title:"My Media",items:[{id:me.MY_MEDIA_ALL,label:"All",status:"ready"},{id:me.MY_MEDIA_VIDEOS,label:"Videos",status:"ready"},{id:me.MY_MEDIA_IMAGES,label:"Images",status:"ready"},{id:me.MY_MEDIA_AUDIO,label:"Audio",status:"ready"},{id:me.MY_MEDIA_PROMPTS,label:"Prompts",status:"ready"}]}],Ex={"image-to-video":"🎬","text-to-video":"📝","text-to-image-to-video":"✨","video-to-video":"🔄","speech-to-video":"🎤","text-to-image":"🖼️","image-to-image":"🎨",upscaler:"🔍",reframe:"📐","face-swap":"🎭","prompt-generator":"💡","image-to-text":"📷","video-to-text":"🎥","audio-generation":"🔊","voice-cloning":"🗣️","lip-sync":"👄",pipeline:"⚙️","lora-training":"🧠","my-media-all":"📁","my-media-videos":"🎞️","my-media-images":"🖼️","my-media-audio":"🎵","my-media-prompts":"📝"};function Ix({activeToolId:c,onSelectTool:x,collapsed:d,onToggleCollapsed:N}){return t.jsxs("aside",{className:`sidebar ${d?"collapsed":""}`,children:[t.jsx("div",{className:"sidebar-header",children:t.jsx("div",{className:"sidebar-logo",children:"Oelala"})}),t.jsx("nav",{className:"sidebar-nav",children:zx.map(p=>t.jsxs("div",{className:"sidebar-group",children:[t.jsx("div",{className:"sidebar-group-title",children:p.title}),p.items.map(b=>{const S=c===b.id,R=Ex[b.id]||"🔧";return t.jsxs("button",{className:`nav-item${S?" active":""}`,onClick:()=>x(b.id),type:"button",children:[t.jsx("span",{className:"nav-icon",style:{fontSize:"16px"},children:R}),t.jsx("span",{className:"nav-label",children:b.label}),b.status==="new"&&t.jsx("span",{className:"nav-badge",children:"new"})]},b.id)})]},p.id))}),t.jsx("div",{className:"sidebar-footer",children:t.jsxs("button",{onClick:N,className:"nav-item collapse-btn",children:[t.jsx("span",{className:"nav-icon",style:{fontSize:"16px"},children:d?"▶️":"◀️"}),t.jsx("span",{className:"nav-label",children:"Collapse"})]})})]})}async function Wa(c){try{await fetch(`${ve}/client-log`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(c)})}catch(x){console.error("Failed to send client log",x)}}function Px(c){const[x,d]=l.useState([]),[N,p]=l.useState(!1),[b,S]=l.useState(""),R=l.useCallback(async()=>{p(!0),S("");try{const I=await fetch(`${ve}/list-videos`),P=await I.json();if(!I.ok)throw new Error(P?.detail||`History failed (${I.status})`);d(Array.isArray(P?.videos)?P.videos:[])}catch(I){const P=I?.message||"Failed to load history";S(P),await Wa({level:"error",message:"History fetch failed",timestamp:new Date().toISOString(),meta:{message:P}})}finally{p(!1)}},[]);return l.useEffect(()=>{R()},[R,c]),{videos:x,loading:N,error:b,refresh:R}}function Wd(c){const x=Math.floor(c/60),d=Math.floor(c%60);return`${x}:${d.toString().padStart(2,"0")}`}function Tx({output:c,refreshToken:x,onSelectHistoryVideo:d,onClose:N}){const[p,b]=l.useState(!1),[S,R]=l.useState(null),[I,P]=l.useState(!1),{videos:T,loading:A,error:C}=Px(x),L=l.useMemo(()=>c?c.kind==="video"?t.jsxs("div",{className:"media-container",children:[t.jsxs("div",{className:"video-wrapper",onMouseEnter:()=>P(!0),onMouseLeave:()=>P(!1),children:[t.jsx("video",{className:"media-preview",controls:!0,src:c.url,autoPlay:!0,loop:!0,onLoadedMetadata:V=>R(V.target.duration)}),I&&S&&t.jsxs("div",{className:"video-duration-overlay",children:[t.jsx(Qn,{size:14}),t.jsx("span",{children:Wd(S)})]})]}),t.jsxs("div",{className:"media-info",children:[t.jsxs("div",{className:"media-meta",children:[c.filename||"Generated Video",S&&t.jsxs("span",{className:"duration-inline",children:[" • ",Wd(S)]})]}),t.jsxs("div",{className:"media-actions",children:[c.url&&t.jsx("a",{className:"icon-btn",href:c.url,download:c.filename||void 0,title:"Download",children:t.jsx(qt,{size:18})}),c.backendUrl&&t.jsx("a",{className:"icon-btn",href:c.backendUrl,target:"_blank",rel:"noreferrer",title:"Open in new tab",children:t.jsx(Ul,{size:18})})]})]})]}):c.kind==="image"?t.jsxs("div",{className:"media-container",children:[t.jsx("img",{className:"media-preview",src:c.url,alt:"Generated",onError:V=>{console.error("Image load failed:",c.url),V.target.style.display="none",V.target.parentNode.innerHTML+=`<div style="padding:20px;color:red">Failed to load image: ${c.url}</div>`}}),t.jsxs("div",{className:"media-info",children:[t.jsx("div",{className:"media-meta",children:c.filename||"Generated Image"}),t.jsxs("div",{className:"media-actions",children:[c.url&&t.jsx("a",{className:"icon-btn",href:c.url,download:c.filename||void 0,title:"Download",children:t.jsx(qt,{size:18})}),c.backendUrl&&t.jsx("a",{className:"icon-btn",href:c.backendUrl,target:"_blank",rel:"noreferrer",title:"Open in new tab",children:t.jsx(Ul,{size:18})})]})]})]}):c.kind==="lora"?t.jsxs("div",{className:"media-container",style:{padding:"24px"},children:[t.jsx("h3",{children:"LoRA Training Complete"}),t.jsxs("div",{className:"media-meta",style:{marginTop:"16px"},children:[t.jsxs("p",{children:["ID: ",c.training_id]}),t.jsxs("p",{children:["Path: ",c.lora_path]})]})]}):null:t.jsxs("div",{className:"placeholder-state",children:[t.jsx("div",{className:"placeholder-icon",children:t.jsx(Ua,{})}),t.jsx("h3",{children:"Ready to Create"}),t.jsx("p",{className:"muted",children:"Configure parameters and click Generate"})]}),[c]);return t.jsxs("section",{className:"output-panel",children:[t.jsxs("div",{style:{position:"absolute",top:20,right:20,zIndex:10,display:"flex",gap:"8px"},children:[t.jsx("button",{className:"icon-btn",onClick:()=>b(!p),title:"History",children:t.jsx(ym,{size:20})}),N&&t.jsx("button",{className:"icon-btn",onClick:N,title:"Close & show My Media",children:t.jsx(lt,{size:20})})]}),L,p&&t.jsxs("div",{className:"history",children:[t.jsxs("div",{className:"history-title",style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[t.jsx("span",{children:"History"}),t.jsx("button",{className:"icon-btn",onClick:()=>b(!1),children:t.jsx(lt,{size:18})})]}),t.jsxs("div",{className:"history-list",children:[A&&t.jsx("div",{style:{padding:20,textAlign:"center"},className:"muted",children:"Loading..."}),C&&t.jsx("div",{className:"error",children:C}),!A&&!C&&T.length===0&&t.jsx("div",{style:{padding:20,textAlign:"center"},className:"muted",children:"No history yet"}),T.map(V=>t.jsxs("button",{className:"history-item",onClick:()=>{d({kind:"video",url:`${ve}/outputs/${V.filename}`,backendUrl:`${ve}/outputs/${V.filename}`,filename:V.filename})},children:[t.jsx("div",{className:"history-item-title",children:V.filename}),t.jsx("div",{className:"history-item-sub",children:new Date(V.mtime*1e3).toLocaleString()})]},V.filename))]})]})]})}function Rx({promptId:c,onComplete:x}){const[d,N]=l.useState(null),[p,b]=l.useState(null),[S,R]=l.useState(0),[I,P]=l.useState(null),[T,A]=l.useState(Date.now()),[C,L]=l.useState(""),V=l.useCallback(async()=>{if(c)try{const j=await fetch(`${ve}/comfyui/job/${c}`);if(!j.ok)return;const k=await j.json();N(k),(k.status==="completed"||k.status==="failed")&&x&&x(k)}catch{}},[c,x]),U=l.useCallback(async()=>{if(c)try{const j=await fetch(`${ve}/comfyui/queue`);if(!j.ok)return;const k=await j.json(),B=k.running.findIndex(v=>v.prompt_id===c),h=k.pending.findIndex(v=>v.prompt_id===c);B>=0?b({status:"running",position:B}):h>=0?b({status:"pending",position:k.running.length+h}):b(null)}catch{}},[c]);if(l.useEffect(()=>{if(S>0&&S<100){const j=Date.now()-T,B=j/S*100-j;P(Math.max(0,Math.round(B/1e3)))}else P(null)},[S,T]),l.useEffect(()=>{if(!c)return;V(),U();const j=setInterval(V,2e3),k=setInterval(U,3e3);return()=>{clearInterval(j),clearInterval(k)}},[c,V,U]),l.useEffect(()=>{d&&(d.status==="completed"?R(100):d.status==="running"?R(j=>Math.min(95,j+5)):d.status==="queued"&&R(0))},[d]),!c||!d)return null;const D=j=>{if(!j||j<=0)return"calculating...";if(j<60)return`${j}s`;const k=Math.floor(j/60),B=j%60;return`${k}m ${B}s`},ee=d.status==="running",Z=d.status==="queued",K=d.status==="completed";return t.jsxs("div",{style:{backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"8px",padding:"16px",marginBottom:"16px"},children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"12px"},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[ee&&t.jsx(at,{size:16,className:"spin",color:"#22c55e"}),Z&&t.jsx(Qn,{size:16,color:"#fbbf24"}),K&&t.jsx(dx,{size:16,color:"#3b82f6"}),t.jsxs("span",{style:{fontWeight:600,fontSize:"0.9rem"},children:[ee&&"Generating...",Z&&"In Queue",K&&"Completed"]})]}),p&&t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"6px",fontSize:"0.75rem",color:"var(--text-muted)",backgroundColor:"var(--bg-input)",padding:"4px 8px",borderRadius:"4px"},children:[t.jsx(Qn,{size:12}),t.jsx("span",{children:p.status==="running"?"Running":`Position: ${p.position+1}`})]})]}),!K&&t.jsx("div",{style:{position:"relative",width:"100%",height:"8px",backgroundColor:"var(--bg-input)",borderRadius:"4px",overflow:"hidden",marginBottom:"8px"},children:t.jsx("div",{style:{position:"absolute",left:0,top:0,height:"100%",width:`${S}%`,backgroundColor:ee?"#22c55e":"#fbbf24",borderRadius:"4px",transition:"width 0.3s ease-out",boxShadow:`0 0 8px ${ee?"rgba(34, 197, 94, 0.5)":"rgba(251, 191, 36, 0.5)"}`}})}),t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",fontSize:"0.75rem",color:"var(--text-muted)"},children:[t.jsxs("span",{children:[ee&&`${S}%`,Z&&"Waiting to start...",K&&"Generation complete"]}),ee&&I!==null&&t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"4px"},children:[t.jsx(Qn,{size:12}),t.jsxs("span",{children:["ETA: ",D(I)]})]})]}),C&&ee&&t.jsx("div",{style:{marginTop:"8px",padding:"6px 8px",backgroundColor:"var(--bg-input)",borderRadius:"4px",fontSize:"0.7rem",color:"var(--text-secondary)"},children:C}),d.preview_url&&t.jsx("div",{style:{marginTop:"12px",borderRadius:"4px",overflow:"hidden",backgroundColor:"var(--bg-input)"},children:t.jsx("img",{src:d.preview_url,alt:"Generation preview",style:{width:"100%",height:"auto",display:"block"}})})]})}function Mx({onJobComplete:c,refreshToken:x}){const[d,N]=l.useState({running:[],pending:[],total_running:0,total_pending:0}),[p,b]=l.useState([]),[S,R]=l.useState(!1),[I,P]=l.useState(new Set),T=l.useRef(null),A=l.useCallback(async()=>{try{const D=await fetch(`${ve}/comfyui/queue`);if(!D.ok)return;const ee=await D.json();N(ee)}catch{}},[]),C=l.useCallback(async D=>{try{const ee=await fetch(`${ve}/comfyui/job/${D}`);return ee.ok?await ee.json():null}catch{return null}},[]);l.useEffect(()=>{A();const D=setInterval(A,3e3);return()=>clearInterval(D)},[A,x]),l.useEffect(()=>{for(const D of p)!I.has(D.prompt_id)&&D.status==="completed"&&D.output_video&&(c&&c(D),P(ee=>new Set([...ee,D.prompt_id])))},[p,I,c]),l.useEffect(()=>{const D=async()=>{for(const ee of d.running){const Z=await C(ee.prompt_id);Z&&Z.status==="completed"&&b(K=>K.some(j=>j.prompt_id===Z.prompt_id)?K:[...K,Z].slice(-10))}};d.running.length>0&&D()},[d.running,C]),l.useEffect(()=>{const D=ee=>{T.current&&!T.current.contains(ee.target)&&R(!1)};if(S)return document.addEventListener("mousedown",D),()=>document.removeEventListener("mousedown",D)},[S]);const L=async D=>{try{await fetch(`${ve}/comfyui/queue/${D}`,{method:"DELETE"}),A()}catch(ee){console.error("Failed to cancel job:",ee)}},V=d.total_running>0,U=d.total_running+d.total_pending;return t.jsxs("div",{style:{position:"relative"},ref:T,children:[t.jsxs("button",{onClick:()=>R(!S),style:{display:"flex",alignItems:"center",gap:"6px",padding:"6px 10px",backgroundColor:V?"rgba(34, 197, 94, 0.15)":"transparent",border:`1px solid ${V?"#22c55e":"var(--border-color)"}`,borderRadius:"6px",cursor:"pointer",color:"var(--text-primary)",fontSize:"0.8rem"},title:V?`${d.total_running} running, ${d.total_pending} queued`:"No active jobs",children:[t.jsx("span",{style:{fontSize:"14px"},children:V?"⏳":"🕐"}),t.jsx("span",{style:{fontWeight:500},children:V?d.total_running:0}),d.total_pending>0&&t.jsxs("span",{style:{color:"var(--text-muted)"},children:["+",d.total_pending]})]}),S&&t.jsxs("div",{style:{position:"absolute",top:"100%",right:0,marginTop:"8px",width:"320px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"8px",boxShadow:"0 4px 20px rgba(0,0,0,0.3)",zIndex:1e3,overflow:"hidden"},children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"10px 12px",borderBottom:"1px solid var(--border-color)",backgroundColor:"var(--bg-primary)"},children:[t.jsx("span",{style:{fontWeight:600,fontSize:"0.85rem"},children:"Generation Queue"}),t.jsxs("div",{style:{display:"flex",gap:"8px"},children:[t.jsx("button",{onClick:A,style:{background:"transparent",border:"none",cursor:"pointer",padding:"2px"},children:t.jsx(_s,{size:12,color:"var(--text-muted)"})}),t.jsx("button",{onClick:()=>R(!1),style:{background:"transparent",border:"none",cursor:"pointer",padding:"2px"},children:t.jsx(lt,{size:14,color:"var(--text-muted)"})})]})]}),t.jsxs("div",{style:{maxHeight:"300px",overflowY:"auto",padding:"8px"},children:[d.running.length>0&&t.jsxs("div",{style:{marginBottom:"8px"},children:[t.jsx("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginBottom:"4px",textTransform:"uppercase"},children:"Running"}),d.running.map(D=>t.jsx(Ml,{job:D,status:"running",onCancel:L},D.prompt_id))]}),d.pending.length>0&&t.jsxs("div",{style:{marginBottom:"8px"},children:[t.jsx("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginBottom:"4px",textTransform:"uppercase"},children:"Pending"}),d.pending.map(D=>t.jsx(Ml,{job:D,status:"pending",onCancel:L},D.prompt_id))]}),p.length>0&&t.jsxs("div",{children:[t.jsx("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginBottom:"4px",textTransform:"uppercase"},children:"Completed"}),p.slice(-3).reverse().map(D=>t.jsx(Ml,{job:D,status:"completed"},D.prompt_id))]}),U===0&&p.length===0&&t.jsx("div",{style:{textAlign:"center",padding:"16px",color:"var(--text-muted)",fontSize:"0.8rem"},children:"No active jobs"})]})]})]})}function Ml({job:c,status:x,onCancel:d}){const[N,p]=l.useState(x==="running"),b={running:"#22c55e",pending:"#fbbf24",completed:"#3b82f6"},S={running:at,pending:Qn,completed:du}[x];return t.jsxs("div",{style:{marginBottom:"4px"},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px",padding:"6px 8px",backgroundColor:"var(--bg-input)",borderRadius:"4px",fontSize:"0.8rem",cursor:x==="running"?"pointer":"default"},onClick:()=>x==="running"&&p(!N),children:[t.jsx(S,{size:12,color:b[x],className:x==="running"?"spin":""}),t.jsxs("div",{style:{flex:1,minWidth:0},children:[t.jsx("div",{style:{whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis",fontWeight:500},children:c.prompt||c.prompt_id.slice(0,8)}),t.jsxs("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)"},children:[c.resolution," ",c.aspect_ratio," ",c.num_frames&&`• ${c.num_frames}f`]})]}),x!=="completed"&&d&&t.jsx("button",{onClick:R=>{R.stopPropagation(),d(c.prompt_id)},style:{background:"transparent",border:"none",cursor:"pointer",padding:"2px"},children:t.jsx(lt,{size:12,color:"var(--text-muted)"})}),x==="completed"&&c.output_video&&t.jsx("a",{href:`${ve}${c.output_video}`,target:"_blank",rel:"noopener noreferrer",style:{color:"#3b82f6",fontSize:"0.7rem"},onClick:R=>R.stopPropagation(),children:"View"})]}),x==="running"&&N&&t.jsx("div",{style:{marginTop:"4px",paddingLeft:"8px"},children:t.jsx(Rx,{promptId:c.prompt_id})})]})}console.warn("⚠️ Supabase credentials not configured. Auth will be disabled.");const Gd=null,gu=l.createContext({user:null,session:null,loading:!0,signInWithGoogle:async()=>{},signInWithGithub:async()=>{},signOut:async()=>{},isAdult:!1,showLoginModal:!1,loginModalMessage:null,requestLogin:()=>{},closeLoginModal:()=>{}});function Lx({children:c}){const[x,d]=l.useState(null),[N,p]=l.useState(null),[b,S]=l.useState(!0),[R,I]=l.useState(!1),[P,T]=l.useState(null);l.useEffect(()=>{{S(!1);return}},[]);const A=async()=>{{console.warn("Auth not enabled");return}},C=async()=>{{console.warn("Auth not enabled");return}},L=async()=>{},V=l.useCallback((Z=null)=>{T(Z),I(!0)},[]),U=l.useCallback(()=>{I(!1),T(null)},[]),ee={user:x,session:N,loading:b,signInWithGoogle:A,signInWithGithub:C,signOut:L,isAdult:!!x,showLoginModal:R,loginModalMessage:P,requestLogin:V,closeLoginModal:U};return t.jsx(gu.Provider,{value:ee,children:c})}function Ke(){const c=l.useContext(gu);if(!c)throw new Error("useAuth must be used within an AuthProvider");return c}async function Jl(){try{if(!Gd)return console.log("🔐 API: supabase client not available"),null;const{data:{session:c},error:x}=await Gd.auth.getSession();return x?(console.error("🔐 API: getSession error:",x),null):(c?.access_token?console.log("🔐 API: Got access token for user:",c.user?.email):console.log("🔐 API: No active session"),c?.access_token||null)}catch(c){return console.error("🔐 API: getAccessToken exception:",c),null}}async function rn(c,x={}){const d=await Jl(),N={...x.headers};d&&(N.Authorization=`Bearer ${d}`),!(x.body instanceof FormData)&&!N["Content-Type"]&&x.body&&(N["Content-Type"]="application/json");const p=c.startsWith("http")?c:`${ve}${c}`;return fetch(p,{...x,headers:N,credentials:"same-origin"})}async function ht(c,x,d={}){const N=await Jl(),p=N?{...d,Authorization:`Bearer ${N}`}:d,b=await fetch(c,{method:"POST",body:x,headers:p,credentials:"same-origin"}),S=await b.text();let R;try{R=S?JSON.parse(S):null}catch{R=S}return b.status===402&&R?.detail&&typeof R.detail=="object"&&R.detail.error==="insufficient_credits"&&typeof R.detail.required=="number"&&typeof R.detail.available=="number"&&window.dispatchEvent(new CustomEvent("insufficient-credits",{detail:{required:R.detail.required,available:R.detail.available,packages:Array.isArray(R.detail.packages)?R.detail.packages:[]}})),{ok:b.ok,status:b.status,data:R}}async function Es(c,x={}){const d=await Jl(),N={"Content-Type":"application/json"};d&&(N.Authorization=`Bearer ${d}`);const p=await fetch(c,{method:"POST",body:JSON.stringify(x),headers:N,credentials:"same-origin"}),b=await p.text();try{const S=b?JSON.parse(b):null;return{ok:p.ok,status:p.status,data:S}}catch{return{ok:p.ok,status:p.status,data:b}}}async function Fx(c){const x=await rn(c,{method:"GET"});if(!x.ok)throw new Error(`API error: ${x.status} ${x.statusText}`);return x.json()}async function Dx(c){const x=await rn(c,{method:"DELETE"});if(!x.ok)throw new Error(`API error: ${x.status} ${x.statusText}`);return x.json()}async function Ox(c="all"){return Fx(`/user/media?type=${c}`)}async function Ax(c,x){return Dx(`/user/media/${c}/${encodeURIComponent(x)}`)}const vu=l.createContext(null);function $x({children:c}){const{user:x}=Ke(),[d,N]=l.useState(0),[p,b]=l.useState(0),[S,R]=l.useState(0),[I,P]=l.useState([]),[T,A]=l.useState(!1),[C,L]=l.useState(null),V=l.useCallback(async()=>{if(!x){N(0),b(0),R(0);return}A(!0),L(null);try{const B=await rn("/api/credits");if(B.ok){const h=await B.json();N(h.balance||0),b(h.lifetime_purchased||0),R(h.lifetime_used||0)}else console.error("Failed to fetch credits:",B.status),N(0)}catch(B){console.error("Credits fetch error:",B),L("Failed to load credits")}finally{A(!1)}},[x]),U=l.useCallback(async()=>{try{const B=await rn("/api/credits/packages");if(B.ok){const h=await B.json();P(h)}}catch(B){console.error("Packages fetch error:",B)}},[]);l.useEffect(()=>{V()},[x?.id]),l.useEffect(()=>{U()},[]);const D=l.useCallback(async(B,h={})=>{try{const v=await rn("/api/credits/estimate",{method:"POST",body:JSON.stringify({generation_type:B,width:h.width||1024,height:h.height||1024,duration_seconds:h.duration_seconds||null,steps:h.steps||20})});if(v.ok)return await v.json()}catch(v){console.error("Estimate error:",v)}return null},[]),ee=l.useCallback(async B=>{if(!x)return L("Please sign in to purchase credits"),null;try{const h=await rn("/api/credits/purchase",{method:"POST",body:JSON.stringify({package_id:B})});if(h.ok)return(await h.json()).checkout_url;{const v=await h.json();return L(v.detail||"Purchase failed"),null}}catch(h){return console.error("Purchase error:",h),L("Purchase failed"),null}},[x]),Z=l.useCallback(B=>d>=B,[d]),K=l.useCallback(B=>{N(h=>Math.max(0,h-B)),R(h=>h+B)},[]),j=l.useCallback(B=>{N(h=>h+B),R(h=>Math.max(0,h-B))},[]),k={balance:d,lifetimePurchased:p,lifetimeUsed:S,packages:I,loading:T,error:C,fetchBalance:V,estimateCost:D,purchaseCredits:ee,hasCredits:Z,deductCredits:K,refundCredits:j,clearError:()=>L(null)};return t.jsx(vu.Provider,{value:k,children:c})}function Zl(){const c=l.useContext(vu);if(!c)throw new Error("useCredits must be used within a CreditsProvider");return c}const Hd=(c,x="EUR")=>new Intl.NumberFormat("nl-NL",{style:"currency",currency:x}).format(c/100),Ux=c=>c==="POPULAR"?{background:"linear-gradient(135deg, #7c3aed, #a855f7)",color:"white"}:c==="BEST VALUE"?{background:"linear-gradient(135deg, #059669, #10b981)",color:"white"}:{background:"#374151",color:"#9ca3af"};function Vx({onClose:c}){const{packages:x,balance:d,purchaseCredits:N,error:p,clearError:b}=Zl(),[S,R]=l.useState(null),[I,P]=l.useState(!1),T=async A=>{R(A.id),P(!0),b();const C=await N(A.id);C?window.location.href=C:(P(!1),R(null))};return t.jsxs(t.Fragment,{children:[t.jsx("div",{style:{position:"fixed",inset:0,background:"rgba(0, 0, 0, 0.7)",backdropFilter:"blur(4px)",zIndex:1100},onClick:c}),t.jsxs("div",{style:{position:"fixed",top:"50%",left:"50%",transform:"translate(-50%, -50%)",width:"90%",maxWidth:600,maxHeight:"90vh",background:"var(--bg-card, #1a1a2e)",borderRadius:16,boxShadow:"0 25px 50px -12px rgba(0, 0, 0, 0.5)",zIndex:1101,overflow:"hidden",display:"flex",flexDirection:"column"},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"20px 24px",borderBottom:"1px solid var(--border-color, #2d2d4a)",background:"linear-gradient(135deg, rgba(124,58,237,0.1), rgba(168,85,247,0.05))"},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:12},children:[t.jsx("div",{style:{width:40,height:40,borderRadius:10,background:"linear-gradient(135deg, #7c3aed, #a855f7)",display:"flex",alignItems:"center",justifyContent:"center"},children:t.jsx($l,{size:20,color:"white"})}),t.jsxs("div",{children:[t.jsx("h2",{style:{margin:0,fontSize:"1.2rem",color:"var(--text-primary, white)"},children:"Buy Credits"}),t.jsxs("p",{style:{margin:0,fontSize:"0.8rem",color:"var(--text-muted, #888)"},children:["Current balance: ",t.jsxs("strong",{style:{color:"#a78bfa"},children:[d," credits"]})]})]})]}),t.jsx("button",{onClick:c,style:{background:"none",border:"none",color:"var(--text-muted, #888)",cursor:"pointer",padding:8,borderRadius:8,transition:"background 0.15s"},children:t.jsx(lt,{size:20})})]}),p&&t.jsx("div",{style:{margin:"16px 24px 0",padding:"12px 16px",background:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.3)",borderRadius:8,color:"#f87171",fontSize:"0.85rem"},children:p}),t.jsxs("div",{style:{padding:"24px",overflowY:"auto",flex:1},children:[t.jsx("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fit, minmax(160px, 1fr))",gap:16},children:x.map(A=>{const C=S===A.id,L=A.price_cents/A.credits,U=Math.round((1-L/100/.05)*100);return t.jsxs("div",{onClick:()=>!I&&T(A),style:{position:"relative",padding:"20px 16px",borderRadius:12,border:C?"2px solid #7c3aed":"1px solid var(--border-color, #2d2d4a)",background:C?"rgba(124, 58, 237, 0.1)":"var(--bg-input, #252540)",cursor:I?"wait":"pointer",transition:"all 0.2s ease",opacity:I&&!C?.5:1},children:[A.badge&&t.jsx("div",{style:{position:"absolute",top:-10,right:12,padding:"4px 10px",borderRadius:20,fontSize:"0.65rem",fontWeight:600,textTransform:"uppercase",letterSpacing:"0.5px",...Ux(A.badge)},children:A.badge}),t.jsx("div",{style:{fontSize:"0.9rem",fontWeight:600,color:"var(--text-primary, white)",marginBottom:8},children:A.name}),t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:6,marginBottom:12},children:[t.jsx(Cn,{size:16,style:{color:"#fbbf24"}}),t.jsx("span",{style:{fontSize:"1.5rem",fontWeight:700,color:"#a78bfa"},children:A.credits.toLocaleString()})]}),t.jsx("div",{style:{fontSize:"1.1rem",fontWeight:600,color:"var(--text-primary, white)",marginBottom:4},children:Hd(A.price_cents,A.currency)}),t.jsxs("div",{style:{fontSize:"0.75rem",color:"var(--text-muted, #888)"},children:[Hd(L,A.currency),"/credit",U>0&&t.jsxs("span",{style:{color:"#10b981",marginLeft:6},children:["Save ",U,"%"]})]}),C&&I&&t.jsx("div",{style:{position:"absolute",inset:0,background:"rgba(0,0,0,0.5)",borderRadius:12,display:"flex",alignItems:"center",justifyContent:"center"},children:t.jsx(at,{size:24,className:"spin",style:{color:"#a78bfa"}})})]},A.id)})}),t.jsx("div",{style:{marginTop:24,padding:"16px",background:"rgba(124, 58, 237, 0.05)",borderRadius:12,border:"1px solid rgba(124, 58, 237, 0.2)"},children:t.jsxs("div",{style:{display:"flex",alignItems:"flex-start",gap:12,fontSize:"0.8rem",color:"var(--text-muted, #888)"},children:[t.jsx(mr,{size:16,style:{color:"#10b981",flexShrink:0,marginTop:2}}),t.jsxs("div",{children:[t.jsx("strong",{style:{color:"var(--text-primary, white)"},children:"Credits never expire."}),t.jsx("br",{}),"Use them whenever you want. Secure payment via Stripe."]})]})})]}),t.jsxs("div",{style:{padding:"16px 24px",borderTop:"1px solid var(--border-color, #2d2d4a)",display:"flex",alignItems:"center",justifyContent:"space-between",fontSize:"0.75rem",color:"var(--text-muted, #888)"},children:[t.jsx("span",{children:"Payments processed securely by Stripe"}),t.jsxs("a",{href:"https://stripe.com",target:"_blank",rel:"noopener noreferrer",style:{display:"flex",alignItems:"center",gap:4,color:"inherit",textDecoration:"none"},children:[t.jsx(Ul,{size:12}),"stripe.com"]})]})]}),t.jsx("style",{children:`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .spin {
          animation: spin 1s linear infinite;
        }
      `})]})}function Bx(){const{user:c,loading:x,signInWithGoogle:d,signOut:N}=Ke(),{balance:p,loading:b}=Zl(),[S,R]=l.useState(!1),[I,P]=l.useState(!1);return x?t.jsx("div",{className:"user-menu loading",children:t.jsx(at,{size:16,className:"spin"})}):c?t.jsxs("div",{className:"user-menu",style:{position:"relative",display:"flex",alignItems:"center",gap:"8px"},children:[t.jsxs("div",{className:"credits-display",onClick:()=>P(!0),style:{display:"flex",alignItems:"center",gap:"4px",padding:"6px 10px",borderRadius:"6px",background:"linear-gradient(135deg, #7c3aed22, #a855f722)",border:"1px solid #7c3aed44",color:"#a78bfa",fontSize:"0.85rem",fontWeight:600,cursor:"pointer",transition:"all 0.2s ease"},title:"Click to buy credits",children:[t.jsx($l,{size:14}),t.jsx("span",{children:b?"...":p}),t.jsx(Vd,{size:12,style:{opacity:.7}})]}),t.jsxs("button",{className:"user-info-btn",onClick:()=>R(!S),title:c.email,style:{display:"flex",alignItems:"center",gap:"6px",padding:"6px 10px",borderRadius:"6px",border:"1px solid var(--border-color)",background:"var(--bg-input)",color:"var(--text-secondary)",fontSize:"0.8rem",cursor:"pointer",transition:"all 0.2s ease"},children:[c.user_metadata?.avatar_url?t.jsx("img",{src:c.user_metadata.avatar_url,alt:"Avatar",className:"user-avatar",style:{width:24,height:24,borderRadius:"50%"}}):t.jsx(Gl,{size:16}),t.jsx("span",{className:"user-name",style:{maxWidth:100,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"},children:c.user_metadata?.full_name||c.email?.split("@")[0]}),t.jsx(Qt,{size:14,style:{opacity:.6}})]}),S&&t.jsxs("div",{className:"user-dropdown",style:{position:"absolute",top:"100%",right:0,marginTop:4,minWidth:200,background:"var(--bg-card)",border:"1px solid var(--border-color)",borderRadius:8,boxShadow:"0 4px 12px rgba(0,0,0,0.3)",zIndex:1e3,overflow:"hidden"},children:[t.jsxs("div",{style:{padding:"12px 14px",borderBottom:"1px solid var(--border-color)"},children:[t.jsx("div",{style:{fontSize:"0.85rem",fontWeight:500,color:"var(--text-primary)"},children:c.user_metadata?.full_name||"User"}),t.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:2},children:c.email})]}),t.jsxs("div",{style:{padding:"12px 14px",borderBottom:"1px solid var(--border-color)",background:"rgba(124,58,237,0.05)"},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:6},children:[t.jsx("span",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Credits Balance"}),t.jsxs("span",{style:{fontSize:"0.95rem",fontWeight:600,color:"#a78bfa"},children:[t.jsx($l,{size:14,style:{marginRight:4,verticalAlign:"middle"}}),p]})]}),t.jsxs("button",{onClick:()=>{P(!0),R(!1)},style:{display:"flex",alignItems:"center",justifyContent:"center",gap:6,width:"100%",padding:"8px 12px",border:"none",borderRadius:6,background:"linear-gradient(135deg, #7c3aed, #a855f7)",color:"white",fontSize:"0.8rem",fontWeight:500,cursor:"pointer",transition:"opacity 0.15s"},onMouseEnter:T=>T.target.style.opacity="0.9",onMouseLeave:T=>T.target.style.opacity="1",children:[t.jsx(Vd,{size:14}),"Buy Credits"]})]}),t.jsxs("button",{onClick:()=>{N(),R(!1)},style:{display:"flex",alignItems:"center",gap:8,width:"100%",padding:"10px 14px",border:"none",background:"transparent",color:"#ef4444",fontSize:"0.85rem",cursor:"pointer",transition:"background 0.15s"},onMouseEnter:T=>T.target.style.background="rgba(239,68,68,0.1)",onMouseLeave:T=>T.target.style.background="transparent",children:[t.jsx(_m,{size:16}),"Sign out"]})]}),S&&t.jsx("div",{style:{position:"fixed",inset:0,zIndex:999},onClick:()=>R(!1)}),I&&t.jsx(Vx,{onClose:()=>P(!1)})]}):t.jsxs("button",{className:"login-btn",onClick:d,title:"Sign in with Google",children:[t.jsx(Nm,{size:16}),t.jsx("span",{children:"Login"})]})}function Wx({onShowLegal:c}){return t.jsxs("footer",{style:{padding:"4px 12px",borderTop:"1px solid rgba(255,255,255,0.03)",background:"transparent",display:"flex",justifyContent:"center",alignItems:"center",gap:"6px",fontSize:"9px",color:"#3b4555"},children:[t.jsxs("span",{children:["© ",new Date().getFullYear()," oelala.xyz"]}),t.jsx("span",{style:{opacity:.3},children:"•"}),t.jsx("button",{onClick:()=>c?.("privacy"),style:$a,children:"Privacy"}),t.jsx("span",{style:{opacity:.3},children:"•"}),t.jsx("button",{onClick:()=>c?.("terms"),style:$a,children:"Terms"}),t.jsx("span",{style:{opacity:.3},children:"•"}),t.jsx("button",{onClick:()=>c?.("dmca"),style:$a,children:"DMCA"})]})}const $a={background:"transparent",border:"none",color:"#3b4555",fontSize:"9px",cursor:"pointer",padding:0};$a[":hover"]={color:"#6b7280"};const Gx=`
# Privacy Policy

**Last updated: January 5, 2026**

oelala.xyz ("we", "our", or "us") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, and share information about you when you use our AI image and video generation service.

## 1. Information We Collect

### Account Information
- Email address (for authentication)
- Display name (optional)

### Generated Content
- Images and videos you generate using our service
- Prompts and settings used for generation
- Generation history and favorites

### Usage Data
- Features used and generation counts
- Credit usage and transaction history
- Device type and browser information

### Payment Information
When you purchase credits, payment processing is handled by Stripe. We do not store your credit card details.

## 2. How We Use Your Information

We use your information to:
- Provide and improve our AI generation service
- Process credit purchases and manage your account
- Send service-related notifications
- Prevent abuse and enforce our Terms of Service
- Comply with legal obligations

## 3. Data Storage

- **Authentication**: Supabase (EU region)
- **Generated Media**: Our own servers (Netherlands)
- **Payments**: Stripe

### Data Retention
| Data Type | Retention Period |
|-----------|------------------|
| Account data | Until account deletion |
| Generated media (Free tier) | 30 days |
| Generated media (Paid tier) | Per subscription terms |

## 4. Your Rights (GDPR)

As an EU-based service, you have the right to:
- **Access**: Request a copy of your data
- **Rectification**: Correct inaccurate data
- **Erasure**: Delete your account and data
- **Portability**: Export your data
- **Object**: Opt out of certain processing

To exercise these rights, email: **privacy@oelala.xyz**

## 5. Cookies

We use essential cookies only for authentication and preferences. We do not use tracking or advertising cookies.

## 6. Contact Us

For privacy-related questions: **privacy@oelala.xyz**
`,Hx=`
# Terms of Service

**Last updated: January 5, 2026**

By using oelala.xyz ("Service"), you agree to these Terms. Please read them carefully.

## 1. Eligibility

You must be at least 18 years old to use this Service.

## 2. Account Security

You are responsible for maintaining the security of your account and for all activities under your account.

## 3. Credits and Payments

- Generations require credits
- Credits are purchased in packages
- Credits are non-refundable except as required by law
- Unused credits may expire per your subscription tier

## 4. Acceptable Use

### You May NOT Generate:
- Child sexual abuse material (CSAM) – **zero tolerance**
- Non-consensual intimate imagery of real people
- Content that infringes intellectual property
- Content promoting violence, terrorism, or hate
- Fraudulent deepfakes

### We Reserve the Right To:
- Remove violating content
- Suspend or terminate accounts
- Report illegal content to authorities

## 5. Intellectual Property

You retain ownership of content you generate, subject to AI model licenses.

## 6. Disclaimer

THE SERVICE IS PROVIDED "AS IS" WITHOUT WARRANTIES. We do not guarantee uninterrupted service or accuracy of AI-generated content.

## 7. Limitation of Liability

Our total liability is limited to the amount you paid us in the past 12 months.

## 8. Governing Law

These Terms are governed by the laws of the Netherlands.

## 9. Contact

For questions: **legal@oelala.xyz**

*By using oelala.xyz, you agree to be bound by these Terms.*
`,qx=`
# DMCA Policy

**Last updated: January 5, 2026**

oelala.xyz respects intellectual property rights. This policy outlines our procedures for handling copyright infringement claims.

## Reporting Copyright Infringement

Send a written notice to **dmca@oelala.xyz** containing:

1. Your contact information
2. Identification of the copyrighted work
3. URL of the infringing content
4. Statement of good faith belief
5. Statement of accuracy (under penalty of perjury)
6. Your signature

## Our Response

Upon receiving a valid DMCA notice, we will:
1. Remove or disable access to the content
2. Notify the user who posted the content
3. Provide opportunity for counter-notification

## Counter-Notification

If you believe your content was removed in error, submit a counter-notification to **dmca@oelala.xyz** with:
1. Your contact information
2. Identification of the removed content
3. Statement of good faith belief
4. Consent to jurisdiction
5. Your signature

## Repeat Infringer Policy

- First offense: Warning and content removal
- Second offense: Account suspension
- Third offense: Permanent termination

## AI-Generated Content Note

AI-generated content may incorporate learned patterns from training data. If you believe content infringes your specific copyrighted work, include detailed information about your original work.
`,qd={privacy:{title:"Privacy Policy",icon:ex,content:Gx},terms:{title:"Terms of Service",icon:Bl,content:Hx},dmca:{title:"DMCA Policy",icon:Gm,content:qx}};function Qx(c){const x=c.trim().split(`
`),d=[];let N=!1,p=[];for(let b=0;b<x.length;b++){const S=x[b];if(S.startsWith("|")){N||(N=!0,p=[]),S.includes("---")||p.push(S.split("|").filter(R=>R.trim()));continue}else N&&(d.push(t.jsxs("table",{style:kn.table,children:[t.jsx("thead",{children:t.jsx("tr",{children:p[0]?.map((R,I)=>t.jsx("th",{style:kn.th,children:R.trim()},I))})}),t.jsx("tbody",{children:p.slice(1).map((R,I)=>t.jsx("tr",{children:R.map((P,T)=>t.jsx("td",{style:kn.td,children:P.trim()},T))},I))})]},`table-${b}`)),N=!1,p=[]);S.startsWith("# ")?d.push(t.jsx("h1",{style:kn.h1,children:S.slice(2)},b)):S.startsWith("## ")?d.push(t.jsx("h2",{style:kn.h2,children:S.slice(3)},b)):S.startsWith("### ")?d.push(t.jsx("h3",{style:kn.h3,children:S.slice(4)},b)):S.startsWith("- ")?d.push(t.jsx("li",{style:kn.li,children:Qd(S.slice(2))},b)):S.startsWith("**")&&S.endsWith("**")?d.push(t.jsx("p",{style:kn.bold,children:S.slice(2,-2)},b)):S.trim()?d.push(t.jsx("p",{style:kn.p,children:Qd(S)},b)):d.push(t.jsx("div",{style:{height:"8px"}},b))}return d}function Qd(c){return c.split(/(\*\*[^*]+\*\*)/g).map((d,N)=>d.startsWith("**")&&d.endsWith("**")?t.jsx("strong",{children:d.slice(2,-2)},N):d)}const kn={h1:{fontSize:"24px",fontWeight:"bold",marginBottom:"16px",color:"#fff"},h2:{fontSize:"18px",fontWeight:"600",marginTop:"24px",marginBottom:"12px",color:"#e5e7eb"},h3:{fontSize:"15px",fontWeight:"600",marginTop:"16px",marginBottom:"8px",color:"#d1d5db"},p:{fontSize:"14px",lineHeight:"1.6",marginBottom:"8px",color:"#9ca3af"},bold:{fontSize:"14px",fontWeight:"600",marginBottom:"16px",color:"#9ca3af"},li:{fontSize:"14px",lineHeight:"1.6",marginLeft:"16px",marginBottom:"4px",color:"#9ca3af"},table:{width:"100%",borderCollapse:"collapse",marginBottom:"16px"},th:{padding:"8px 12px",textAlign:"left",borderBottom:"1px solid #374151",color:"#e5e7eb",fontSize:"13px"},td:{padding:"8px 12px",borderBottom:"1px solid #1f2937",color:"#9ca3af",fontSize:"13px"}};function Yx({type:c="privacy",onClose:x}){const d=qd[c]||qd.privacy,N=d.icon;return l.useEffect(()=>{const p=b=>{b.key==="Escape"&&x()};return window.addEventListener("keydown",p),()=>window.removeEventListener("keydown",p)},[x]),t.jsxs(t.Fragment,{children:[t.jsx("div",{onClick:x,style:{position:"fixed",inset:0,background:"rgba(0, 0, 0, 0.75)",zIndex:9998}}),t.jsxs("div",{style:{position:"fixed",top:"50%",left:"50%",transform:"translate(-50%, -50%)",width:"90%",maxWidth:"700px",maxHeight:"85vh",background:"#111827",borderRadius:"12px",border:"1px solid #374151",zIndex:9999,display:"flex",flexDirection:"column",overflow:"hidden"},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"16px 20px",borderBottom:"1px solid #374151"},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"10px"},children:[t.jsx(N,{size:20,style:{color:"#8b5cf6"}}),t.jsx("span",{style:{color:"#fff",fontWeight:"600",fontSize:"16px"},children:d.title})]}),t.jsx("button",{onClick:x,style:{background:"transparent",border:"none",cursor:"pointer",padding:"4px",display:"flex"},children:t.jsx(lt,{size:20,style:{color:"#6b7280"}})})]}),t.jsx("div",{style:{flex:1,overflow:"auto",padding:"20px"},children:Qx(d.content)}),t.jsxs("div",{style:{padding:"12px 20px",borderTop:"1px solid #374151",display:"flex",justifyContent:"space-between",alignItems:"center"},children:[t.jsxs("div",{style:{display:"flex",gap:"16px"},children:[c!=="privacy"&&t.jsx("button",{onClick:()=>window.dispatchEvent(new CustomEvent("showLegal",{detail:"privacy"})),style:Ll,children:"Privacy"}),c!=="terms"&&t.jsx("button",{onClick:()=>window.dispatchEvent(new CustomEvent("showLegal",{detail:"terms"})),style:Ll,children:"Terms"}),c!=="dmca"&&t.jsx("button",{onClick:()=>window.dispatchEvent(new CustomEvent("showLegal",{detail:"dmca"})),style:Ll,children:"DMCA"})]}),t.jsx("button",{onClick:x,style:{background:"#374151",color:"#fff",border:"none",borderRadius:"6px",padding:"8px 16px",fontSize:"14px",cursor:"pointer"},children:"Close"})]})]})]})}const Ll={background:"transparent",border:"none",color:"#8b5cf6",fontSize:"13px",cursor:"pointer",textDecoration:"underline"};function Xx({onClose:c,message:x}){const{signInWithGoogle:d,signInWithGithub:N}=Ke(),p=async()=>{await d(),c()},b=async()=>{await N(),c()};return t.jsx("div",{className:"login-modal-overlay",onClick:c,children:t.jsxs("div",{className:"login-modal",onClick:S=>S.stopPropagation(),children:[t.jsx("button",{className:"login-modal-close",onClick:c,children:t.jsx(lt,{size:20})}),t.jsxs("div",{className:"login-modal-content",children:[t.jsx("div",{className:"login-modal-logo",children:t.jsx("h2",{children:"🎬 Oelala"})}),t.jsx("p",{className:"login-modal-message",children:x||"Log in om door te gaan"}),t.jsxs("div",{className:"login-modal-buttons",children:[t.jsxs("button",{className:"login-btn google",onClick:p,children:[t.jsxs("svg",{viewBox:"0 0 24 24",width:"20",height:"20",children:[t.jsx("path",{fill:"#4285F4",d:"M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"}),t.jsx("path",{fill:"#34A853",d:"M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"}),t.jsx("path",{fill:"#FBBC05",d:"M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"}),t.jsx("path",{fill:"#EA4335",d:"M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"})]}),"Continue with Google"]}),t.jsxs("button",{className:"login-btn github",onClick:b,children:[t.jsx("svg",{viewBox:"0 0 24 24",width:"20",height:"20",fill:"currentColor",children:t.jsx("path",{d:"M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"})}),"Continue with GitHub"]})]}),t.jsx("p",{className:"login-modal-note",children:"Je kunt de site bekijken zonder account, maar voor video generatie is inloggen vereist."})]})]})})}const yu=l.createContext({nsfwEnabled:!1,setNsfwEnabled:()=>{}}),Yd="oelala_nsfw_enabled";function Kx({children:c}){const{user:x}=Ke(),[d,N]=l.useState(()=>{try{return localStorage.getItem(Yd)==="true"}catch{return!1}});l.useEffect(()=>{!x&&d&&N(!1)},[x,d]);const p=S=>{S&&!x||N(S)};l.useEffect(()=>{try{x&&localStorage.setItem(Yd,d.toString())}catch{}},[d,x]);const b=x?d:!1;return t.jsx(yu.Provider,{value:{nsfwEnabled:b,setNsfwEnabled:p},children:c})}function Ga(){return l.useContext(yu)}const ql=[{value:"",label:"None",desc:"No camera motion",prefix:""},{value:"static",label:"📷 Static",desc:"Camera stays still",prefix:"static camera shot, "},{value:"pan_left",label:"⬅️ Pan Left",desc:"Camera pans left",prefix:"camera slowly panning left, "},{value:"pan_right",label:"➡️ Pan Right",desc:"Camera pans right",prefix:"camera slowly panning right, "},{value:"tilt_up",label:"⬆️ Tilt Up",desc:"Camera tilts up",prefix:"camera slowly tilting up, "},{value:"tilt_down",label:"⬇️ Tilt Down",desc:"Camera tilts down",prefix:"camera slowly tilting down, "},{value:"zoom_in",label:"🔍 Zoom In",desc:"Camera zooms in",prefix:"camera slowly zooming in, "},{value:"zoom_out",label:"🔭 Zoom Out",desc:"Camera zooms out",prefix:"camera slowly zooming out, "},{value:"dolly_in",label:"🎬 Dolly In",desc:"Camera moves forward",prefix:"camera dollying forward, "},{value:"dolly_out",label:"🎬 Dolly Out",desc:"Camera moves back",prefix:"camera dollying backward, "},{value:"orbit_left",label:"🔄 Orbit Left",desc:"Camera orbits left",prefix:"camera orbiting left around subject, "},{value:"orbit_right",label:"🔄 Orbit Right",desc:"Camera orbits right",prefix:"camera orbiting right around subject, "},{value:"handheld",label:"📹 Handheld",desc:"Slight shake",prefix:"shaky handheld camera, "},{value:"tracking",label:"🏃 Tracking",desc:"Follows subject",prefix:"camera tracking shot following subject, "},{value:"crane_up",label:"🏗️ Crane Up",desc:"Camera rises up",prefix:"crane shot rising up, "},{value:"crane_down",label:"🏗️ Crane Down",desc:"Camera lowers",prefix:"crane shot lowering down, "}];function bu(c){return ql.find(d=>d.value===c)?.prefix||""}function ju({value:c,onChange:x,style:d={}}){const N=ql.find(p=>p.value===c);return t.jsxs("div",{style:{marginBottom:"12px",...d},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px",marginBottom:"6px"},children:[t.jsx("span",{style:{fontSize:"0.85rem",color:"var(--text-secondary)"},children:"Camera Motion"}),t.jsx("span",{style:{fontSize:"0.7rem",color:"var(--text-muted)"},children:c?N?.desc:"Optional"})]}),t.jsx("div",{style:{display:"flex",flexWrap:"wrap",gap:"6px"},children:ql.map(p=>t.jsx("button",{onClick:()=>x(p.value===c?"":p.value),type:"button",style:{padding:"6px 10px",borderRadius:"6px",border:c===p.value?"1px solid var(--accent-color)":"1px solid var(--border-color)",background:c===p.value?"rgba(59, 130, 246, 0.2)":"rgba(255,255,255,0.05)",color:c===p.value?"var(--accent-color)":"var(--text-secondary)",fontSize:"0.8rem",cursor:"pointer",transition:"all 0.15s ease"},title:p.desc,children:p.label},p.value))})]})}const Xd=["Gentle waves lapping on a tropical beach at sunset, palm trees swaying softly in the breeze, golden hour light reflecting on the water","A serene mountain lake with crystal clear water, surrounded by snow-capped peaks, subtle ripples from a gentle breeze","Cherry blossoms falling gracefully in a Japanese garden, petals dancing in the wind, soft spring sunlight","Northern lights dancing across an Arctic sky, vibrant greens and purples, stars twinkling in the background","A misty forest at dawn, sunbeams breaking through the canopy, dew drops glistening on leaves","A cozy coffee shop window on a rainy day, raindrops sliding down the glass, warm light from inside","Tokyo cityscape at night, neon signs flickering, light trails from passing cars","A quaint European village street, autumn leaves blowing, warm café lights in windows","Modern architecture with flowing water features, reflections dancing on glass surfaces","Colorful ink drops spreading in water, hypnotic swirling patterns, smooth organic motion","Floating soap bubbles catching rainbow light, drifting slowly through a sunlit room","Abstract fluid art in motion, vibrant colors mixing and flowing, mesmerizing patterns","Geometric shapes morphing and transforming, satisfying transitions, clean minimalist style","A skilled barista pouring latte art, steam rising from the cup, focused concentration","A dancer gracefully spinning, flowing fabric catching the light, elegant movement","An artist painting on a large canvas, bold brushstrokes, creative energy","A chef plating an exquisite dish, precise movements, steam rising from the food","A majestic eagle soaring through mountain clouds, wings spread wide, powerful and free","Colorful koi fish swimming in a crystal clear pond, graceful movements, dappled sunlight","A fluffy cat stretching lazily by a sunny window, content and peaceful","Butterflies dancing around a blooming flower garden, vibrant colors, gentle flight","A magical portal opening with swirling energy, mystical light emanating, otherworldly glow","A futuristic cityscape with flying vehicles, holographic advertisements, sleek architecture","An enchanted forest with glowing mushrooms, fireflies dancing, magical atmosphere","A steampunk clockwork mechanism turning, brass gears rotating, intricate details"],Jx=[];function ei(c=!1){const x=c?[...Xd,...Jx]:Xd,d=Math.floor(Math.random()*x.length);return x[d]}function wu(c=!1){try{const x=localStorage.getItem("oelala_last_prompt");if(x&&x.trim())return x}catch{}return ei(c)}const Zx=90,eh={"360p":.6,"480p":1,"540p":1.3,"720p":2,"1080p":4},th=15;function ku({resolution:c="480p",duration:x=6,steps:d=6}){let N=Zx;const p=eh[c]||1;N*=p,N+=(x-6)*th,d>6&&(N*=d/6);const b=Math.round(N*.8),S=Math.round(N*1.3);return{seconds:Math.round(N),min:b,max:S,formatted:Vr(N),range:`${Vr(b)} - ${Vr(S)}`}}function nh({resolution:c="480p",numFrames:x=41,steps:d=6,t2iSteps:N=20}){const p=x/16,b=N*1.5;let S=ku({resolution:c,duration:p,steps:d});return S.seconds+=b,S.min+=b*.8,S.max+=b*1.2,S.formatted=Vr(S.seconds),S.range=`${Vr(S.min)} - ${Vr(S.max)}`,S}function Vr(c){if(c<60)return`${Math.round(c)}s`;const x=Math.floor(c/60),d=Math.round(c%60);return d===0?`${x} min`:`${x}m ${d}s`}const rh=[{value:"480p",label:"480p",desc:"Fast"},{value:"720p",label:"720p",desc:"Balanced"}],sh=[8,12,16,24],ah=["16:9","9:16","1:1","4:3","3:4"];function oh({onOutput:c,onRefreshHistory:x,onJobSubmitted:d}){const{user:N,requestLogin:p}=Ke(),[b,S]=l.useState(()=>{const F=localStorage.getItem("t2v_prompt");return F&&F.trim()?F:wu(!1)}),[R,I]=l.useState("blurry, low quality, distorted, ugly"),[P,T]=l.useState(41),[A,C]=l.useState("1:1"),[L,V]=l.useState("480p"),[U,D]=l.useState(16),[ee,Z]=l.useState(""),[K,j]=l.useState(!1),[k,B]=l.useState(6),[h,v]=l.useState(1),[te,re]=l.useState(-1),[xe,ge]=l.useState(20),[E,ue]=l.useState(6),[fe,ie]=l.useState(!1),[W,G]=l.useState(""),[X,J]=l.useState(null),m=F=>{S(F),localStorage.setItem("t2v_prompt",F)},$=l.useMemo(()=>b.trim().length>0&&!fe,[b,fe]),q=l.useMemo(()=>nh({resolution:L,numFrames:P,steps:k,t2iSteps:xe}),[L,P,k,xe]),le=async()=>{if(!N){p("Log in om te genereren");return}if(!b.trim()){G("Prompt is required");return}ie(!0),G(""),J(null);const _=bu(ee)+b,Y=new FormData;Y.append("prompt",_),Y.append("num_frames",String(P)),Y.append("aspect_ratio",A),Y.append("resolution",L),Y.append("fps",String(U));try{const Q=await ht(`${ve}/generate-text`,Y);if(!Q.ok)throw new Error(Q.data?.detail||`Generation failed (status ${Q.status})`);const u=Q.data?.prompt_id;if(!u)throw new Error("No prompt_id returned");J({promptId:u,prompt:b.substring(0,40)+(b.length>40?"...":"")}),d&&d({prompt_id:u})}catch(Q){const u=Q?.message||"Failed to generate video";G(u),await Wa({level:"error",message:"Text-to-video failed",timestamp:new Date().toISOString(),meta:{message:u}})}finally{ie(!1)}};return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{style:{display:"flex",alignItems:"center",justifyContent:"space-between"},children:[t.jsxs("span",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[t.jsx(Xn,{size:18}),"Video Prompt"]}),t.jsx("button",{className:"icon-btn",style:{width:"28px",height:"28px",fontSize:"16px"},onClick:()=>m(ei(!1)),title:"Generate random creative prompt",children:"✨"})]}),t.jsx("textarea",{className:"prompt-textarea",value:b,onChange:F=>m(F.target.value),rows:4,placeholder:"Describe the video you want to generate... (e.g., 'a cat walking through a field of flowers, cinematic')"}),t.jsxs("div",{className:"char-count",children:[b.length," characters"]}),t.jsx(ju,{value:ee,onChange:Z,style:{marginTop:"12px"}})]}),t.jsxs("div",{className:"tool-section",children:[t.jsx("h3",{children:"Settings"}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Resolution"}),t.jsx("div",{className:"button-group",children:rh.map(F=>t.jsx("button",{className:`btn-option ${L===F.value?"active":""}`,onClick:()=>V(F.value),type:"button",children:F.label},F.value))})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Aspect Ratio"}),t.jsx("div",{className:"button-group",children:ah.map(F=>t.jsx("button",{className:`btn-option ${A===F?"active":""}`,onClick:()=>C(F),type:"button",children:F},F))})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Frame Rate"}),t.jsx("div",{className:"button-group",children:sh.map(F=>t.jsxs("button",{className:`btn-option ${U===F?"active":""}`,onClick:()=>D(F),type:"button",children:[F," fps"]},F))})]}),t.jsxs("div",{className:"form-group",children:[t.jsxs("label",{children:["Duration",t.jsxs("span",{className:"label-value",children:[(P/U).toFixed(1),"s (",P," frames)"]})]}),t.jsx("input",{type:"range",min:"17",max:"81",step:"4",value:P,onChange:F=>T(parseInt(F.target.value,10))}),t.jsxs("div",{className:"range-labels",children:[t.jsxs("span",{children:[(17/U).toFixed(1),"s"]}),t.jsxs("span",{children:[(81/U).toFixed(1),"s"]})]})]})]}),t.jsxs("div",{className:"tool-section collapsible",children:[t.jsxs("button",{className:"section-toggle",onClick:()=>j(!K),children:[t.jsx(Yn,{size:16}),"Advanced Settings",t.jsx(Qt,{size:16,className:K?"rotated":""})]}),K&&t.jsxs("div",{className:"advanced-content",children:[t.jsxs("div",{className:"form-row",children:[t.jsxs("div",{className:"form-group half",children:[t.jsx("label",{children:"Video Steps"}),t.jsx("input",{type:"number",value:k,onChange:F=>B(parseInt(F.target.value)||6),min:"1",max:"30"})]}),t.jsxs("div",{className:"form-group half",children:[t.jsx("label",{children:"Video CFG"}),t.jsx("input",{type:"number",value:h,onChange:F=>v(parseFloat(F.target.value)||1),min:"0.1",max:"10",step:"0.1"})]})]}),t.jsxs("div",{className:"form-row",children:[t.jsxs("div",{className:"form-group half",children:[t.jsx("label",{children:"T2I Steps"}),t.jsx("input",{type:"number",value:xe,onChange:F=>ge(parseInt(F.target.value)||20),min:"1",max:"50"})]}),t.jsxs("div",{className:"form-group half",children:[t.jsx("label",{children:"T2I CFG"}),t.jsx("input",{type:"number",value:E,onChange:F=>ue(parseFloat(F.target.value)||6),min:"1",max:"20",step:"0.5"})]})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Seed (-1 = random)"}),t.jsx("input",{type:"number",value:te,onChange:F=>re(parseInt(F.target.value)||-1)})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Negative Prompt"}),t.jsx("textarea",{value:R,onChange:F=>I(F.target.value),rows:2,placeholder:"Things to avoid..."})]})]})]}),X&&t.jsx("div",{className:"queued-notice",children:"✅ Job queued! Check the Queue panel for progress."}),W&&t.jsxs("div",{className:"error-message",children:["⚠️ ",W]}),!fe&&$&&t.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"center",gap:"6px",marginBottom:"8px",fontSize:"0.85rem",color:"var(--text-muted)"},children:[t.jsx(Qn,{size:14}),t.jsxs("span",{children:["Estimated time: ~",q.formatted]})]}),t.jsx("button",{className:"btn-primary btn-large",type:"button",disabled:!$,onClick:le,children:fe?t.jsxs(t.Fragment,{children:[t.jsx(at,{size:18,className:"spin"}),"Queueing..."]}):t.jsxs(t.Fragment,{children:[t.jsx(Xn,{size:18}),"Generate Video"]})}),t.jsx("div",{className:"tool-info",children:"💡 Text-to-Video first generates an image from your prompt, then animates it using Wan2.2"}),t.jsx("style",{children:`
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
      `})]})}function lh({onPresetChange:c,onParametersChange:x,currentParameters:d}){const[N,p]=l.useState([]),[b,S]=l.useState(null),[R,I]=l.useState({}),[P,T]=l.useState(!0),[A,C]=l.useState(!0),[L,V]=l.useState(null);l.useEffect(()=>{U()},[]);const U=async()=>{try{C(!0);const h=await fetch(`${ve}/api/presets`);if(!h.ok)throw new Error("Failed to fetch presets");const v=await h.json();if(p(v.presets||[]),v.presets?.length>0){const te=v.presets[0];S(te),D(te)}}catch(h){console.error("Failed to load presets:",h),V(h.message),p(ih())}finally{C(!1)}},D=h=>{if(!h?.parameters)return;const v={};Object.entries(h.parameters).forEach(([te,re])=>{re.type!=="image"&&(v[te]=re.default??re.value??"")}),I(v),x?.(v)},ee=h=>{S(h),D(h),c?.(h)},Z=(h,v,te)=>{let re=v;te.type==="integer"?re=parseInt(v,10):te.type==="float"&&(re=parseFloat(v));const xe={...R,[h]:re};I(xe),x?.(xe)},K=h=>{switch(h){case"ImageToVideo":return t.jsx(Ua,{size:16});case"TextToVideo":return t.jsx(Cn,{size:16});case"TextToImage":return t.jsx(Kl,{size:16});default:return t.jsx(Yn,{size:16})}},j=h=>h.name?.toLowerCase().includes("lightning")||h.name?.toLowerCase().includes("fast")?t.jsx("span",{className:"preset-badge fast",children:"⚡ Fast"}):h.name?.toLowerCase().includes("quality")||h.name?.toLowerCase().includes("q6")?t.jsx("span",{className:"preset-badge quality",children:"💎 Quality"}):h.name?.toLowerCase().includes("nsfw")||h.name?.toLowerCase().includes("enhanced")?t.jsx("span",{className:"preset-badge nsfw",children:"🔥 Enhanced"}):null,k=(h,v)=>{const te=R[h]??v.default??"";return v.type==="image"?null:v.type==="string"?t.jsxs("div",{className:"param-group",children:[t.jsxs("label",{htmlFor:`param-${h}`,children:[v.label||h,v.description&&t.jsx("span",{className:"param-hint",title:v.description,children:"ℹ️"})]}),t.jsx("textarea",{id:`param-${h}`,value:te,onChange:re=>Z(h,re.target.value,v),placeholder:v.description,rows:h.includes("prompt")?3:1})]},h):v.type==="integer"&&v.min!==void 0&&v.max!==void 0?t.jsxs("div",{className:"param-group",children:[t.jsxs("label",{htmlFor:`param-${h}`,children:[v.label||h,": ",t.jsx("span",{className:"param-value",children:te}),v.description&&t.jsx("span",{className:"param-hint",title:v.description,children:"ℹ️"})]}),t.jsx("input",{id:`param-${h}`,type:"range",min:v.min,max:v.max,step:v.step||1,value:te,onChange:re=>Z(h,re.target.value,v)}),t.jsxs("div",{className:"range-labels",children:[t.jsx("span",{children:v.min}),t.jsx("span",{children:v.max})]})]},h):v.type==="float"&&v.min!==void 0&&v.max!==void 0?t.jsxs("div",{className:"param-group",children:[t.jsxs("label",{htmlFor:`param-${h}`,children:[v.label||h,": ",t.jsx("span",{className:"param-value",children:te.toFixed?.(2)||te}),v.description&&t.jsx("span",{className:"param-hint",title:v.description,children:"ℹ️"})]}),t.jsx("input",{id:`param-${h}`,type:"range",min:v.min,max:v.max,step:v.step||.1,value:te,onChange:re=>Z(h,re.target.value,v)}),t.jsxs("div",{className:"range-labels",children:[t.jsx("span",{children:v.min}),t.jsx("span",{children:v.max})]})]},h):v.type==="integer"||v.type==="float"?t.jsxs("div",{className:"param-group",children:[t.jsxs("label",{htmlFor:`param-${h}`,children:[v.label||h,v.description&&t.jsx("span",{className:"param-hint",title:v.description,children:"ℹ️"})]}),t.jsx("input",{id:`param-${h}`,type:"number",value:te,onChange:re=>Z(h,re.target.value,v),step:v.step||(v.type==="float"?.1:1)})]},h):v.type==="boolean"?t.jsx("div",{className:"param-group checkbox",children:t.jsxs("label",{htmlFor:`param-${h}`,children:[t.jsx("input",{id:`param-${h}`,type:"checkbox",checked:!!te,onChange:re=>Z(h,re.target.checked,v)}),v.label||h,v.description&&t.jsx("span",{className:"param-hint",title:v.description,children:"ℹ️"})]})},h):null},B=()=>{if(!b?.parameters)return{};const h={prompt:[],generation:[],dimensions:[],other:[]};return Object.entries(b.parameters).forEach(([v,te])=>{te.type!=="image"&&(v.includes("prompt")?h.prompt.push([v,te]):["steps","cfg","seed","frame_rate"].includes(v)?h.generation.push([v,te]):["width","height","num_frames"].includes(v)?h.dimensions.push([v,te]):h.other.push([v,te]))}),h};return A?t.jsxs("div",{className:"preset-selector loading",children:[t.jsx(zs,{className:"spinning",size:24}),t.jsx("span",{children:"Loading presets..."})]}):t.jsxs("div",{className:"preset-selector",children:[t.jsxs("div",{className:"preset-header",onClick:()=>T(!P),children:[t.jsxs("div",{className:"preset-title",children:[t.jsx(zs,{size:20}),t.jsx("span",{children:"Workflow Preset"}),b&&t.jsx("span",{className:"selected-preset-name",children:b.name})]}),P?t.jsx(Gf,{size:20}):t.jsx(Qt,{size:20})]}),P&&t.jsxs("div",{className:"preset-content",children:[t.jsx("div",{className:"preset-list",children:N.map(h=>t.jsxs("div",{className:`preset-card ${b?.id===h.id?"selected":""}`,onClick:()=>ee(h),children:[t.jsxs("div",{className:"preset-card-header",children:[K(h.category),t.jsx("span",{className:"preset-name",children:h.name}),j(h)]}),t.jsx("p",{className:"preset-description",children:h.description})]},h.id))}),b&&t.jsxs("div",{className:"preset-parameters",children:[t.jsxs("h4",{children:[t.jsx(Yn,{size:16})," Parameters"]}),B().prompt?.length>0&&t.jsxs("div",{className:"param-section",children:[t.jsx("h5",{children:"📝 Prompts"}),B().prompt.map(([h,v])=>k(h,v))]}),B().generation?.length>0&&t.jsxs("div",{className:"param-section",children:[t.jsx("h5",{children:"⚙️ Generation"}),t.jsx("div",{className:"param-grid",children:B().generation.map(([h,v])=>k(h,v))})]}),B().dimensions?.length>0&&t.jsxs("div",{className:"param-section",children:[t.jsx("h5",{children:"📐 Dimensions"}),t.jsx("div",{className:"param-grid",children:B().dimensions.map(([h,v])=>k(h,v))})]}),B().other?.length>0&&t.jsxs("div",{className:"param-section",children:[t.jsx("h5",{children:"🔧 Other"}),B().other.map(([h,v])=>k(h,v))]})]})]}),L&&t.jsxs("div",{className:"preset-error",children:["⚠️ ",L," - Using default presets"]})]})}function ih(){return[{id:"wan22_enhanced_q4km",name:"WAN 2.2 Enhanced NSFW FastMove",category:"ImageToVideo",description:"Lightning-fast I2V with NSFW FastMove LoRAs. 4 steps, cfg=1.",parameters:{prompt:{type:"string",default:"motion, smooth camera movement",label:"Prompt"},steps:{type:"integer",default:4,min:2,max:12,label:"Steps"},cfg:{type:"float",default:1,min:1,max:3,step:.1,label:"CFG Scale"},seed:{type:"integer",default:-1,label:"Seed",description:"-1 for random"},width:{type:"integer",default:480,min:256,max:1280,step:16,label:"Width"},height:{type:"integer",default:480,min:256,max:1280,step:16,label:"Height"},num_frames:{type:"integer",default:41,min:17,max:81,step:8,label:"Frames"}}},{id:"wan22_q6_quality",name:"WAN 2.2 Q6 Quality",category:"ImageToVideo",description:"Higher quality 6-bit model with DPM++ scheduler. Best visual quality.",parameters:{prompt:{type:"string",default:"cinematic motion",label:"Prompt"},steps:{type:"integer",default:8,min:4,max:20,label:"Steps"},cfg:{type:"float",default:2.5,min:1,max:5,step:.1,label:"CFG Scale"},seed:{type:"integer",default:-1,label:"Seed"},width:{type:"integer",default:512,min:256,max:1280,step:16,label:"Width"},height:{type:"integer",default:512,min:256,max:1280,step:16,label:"Height"},num_frames:{type:"integer",default:49,min:17,max:97,step:8,label:"Frames"}}}]}const ch=[8,12,16,24],dh=[{value:"wan2.2",label:"🎬 Wan2.2 14B Q6 DisTorch2",desc:"High quality via ComfyUI"}],uh={"480p":{label:"480p",dimensions:{"16:9":"848×480","9:16":"480×848","1:1":"480×480","4:3":"640×480","3:4":"480×640"}},"576p":{label:"576p",dimensions:{"16:9":"1024×576","9:16":"576×1024","1:1":"576×576","4:3":"768×576","3:4":"576×768"}},"720p":{label:"720p",dimensions:{"16:9":"1280×720","9:16":"720×1280","1:1":"720×720","4:3":"960×720","3:4":"720×960"}}},ph=["16:9","9:16","1:1","4:3","3:4"];function fh({onOutput:c,onRefreshHistory:x,onCreationsModeChange:d,onParamsChange:N,onJobSubmitted:p}){const{nsfwEnabled:b}=Ga(),{user:S,requestLogin:R}=Ke(),I=l.useRef(null),[P,T]=l.useState(null),[A,C]=l.useState(""),[L,V]=l.useState("file"),[U,D]=l.useState(()=>wu(!1)),[ee,Z]=l.useState("low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, unnatural sand behavior, visual glitches"),[K,j]=l.useState(!1),[k,B]=l.useState(!1),[h,v]=l.useState(6),[te,re]=l.useState("480p"),[xe,ge]=l.useState("wan2.2"),[E,ue]=l.useState("v2"),[fe,ie]=l.useState(!1),[W,G]=l.useState("9:16"),[X,J]=l.useState(16),[m,$]=l.useState(6),[q,le]=l.useState(1),[F,_]=l.useState(-1),[Y,Q]=l.useState(!1),[u,he]=l.useState(""),[ze,pe]=l.useState({high_noise:[],low_noise:[],general:[]}),[Ne,Re]=l.useState([]),[Ve,it]=l.useState(!1),[Je,Kn]=l.useState({high_noise:[],low_noise:[],pairs:[]}),[_t,We]=l.useState("wan2.2_i2v_high_noise_14B_Q6_K.gguf"),[kt,_n]=l.useState("wan2.2_i2v_low_noise_14B_Q6_K.gguf"),[pn,hr]=l.useState(!1),[zn,gr]=l.useState(!1),[Mt,fn]=l.useState(1),[mn,$t]=l.useState(!1),[Fe,Wr]=l.useState(null),[w,ce]=l.useState({}),[ae,ke]=l.useState(!1),[Ae,Oe]=l.useState(""),[ct,St]=l.useState(null),Qe=l.useMemo(()=>!!P&&!ae,[P,ae]),Ge=l.useMemo(()=>ku({resolution:te,duration:h,steps:m}),[te,h,m]);l.useEffect(()=>{(async()=>{try{const be=await fetch(`${ve}/loras`);if(be.ok){const Te=await be.json();pe(Te)}}catch(be){console.error("Failed to fetch LoRAs:",be)}})()},[]);const Lt=l.useMemo(()=>{if(b)return ze;const H=Te=>(Te||[]).filter(Le=>!Le.nsfw),be={};return ze.by_category&&Object.keys(ze.by_category).forEach(Te=>{const Le=H(ze.by_category[Te]);Le.length>0&&(be[Te]=Le)}),{high_noise:H(ze.high_noise),low_noise:H(ze.low_noise),general:H(ze.general),loras:H(ze.loras),by_category:be}},[ze,b]);l.useEffect(()=>{(async()=>{try{const be=await fetch(`${ve}/unet-models`);if(be.ok){const Te=await be.json();Kn(Te)}}catch(be){console.error("Failed to fetch Unet models:",be)}})()},[]),l.useEffect(()=>{if(U)try{localStorage.setItem("oelala_last_prompt",U)}catch{}},[U]),l.useEffect(()=>{N&&N({tool:"ImageToVideo",prompt:U,duration:h,resolution:te,modelMode:xe,modelVersion:E,aspectRatio:W,fps:X,steps:m,cfg:q,seed:F,usePose:fe,loraConfigs:Ne,filename:P?.name||null})},[U,h,te,xe,E,W,X,m,q,F,fe,Ne,P,N]);const Gr=l.useCallback(async H=>{St(H),Oe("");try{const be=`${ve}${H.url}`,Le=await(await fetch(be)).blob(),He=H.filename||H.url.split("/").pop(),Yt=new File([Le],He,{type:Le.type||"image/png"});T(Yt),C(be),V("file"),c({kind:"image",url:be,backendUrl:be,filename:He,meta:{source:"my-creations",originalItem:H}})}catch(be){Oe("Failed to load selected image"),console.error("Error selecting creation:",be)}},[c]);l.useEffect(()=>(d&&d(L==="creations"&&!P,Gr),()=>{d&&d(!1,null)}),[L,P,d,Gr]);const Hr=async H=>{if(!H)return;T(H),Oe(""),St(null);const be=URL.createObjectURL(H);C(be);try{const Te=new FormData;Te.append("file",H);const Le=await fetch(`${ve}/extract-metadata`,{method:"POST",body:Te});if(Le.ok){const He=await Le.json();He.prompt&&!U&&D(He.prompt),He.negative_prompt&&ee==="low quality, blurry, out of focus, unstable camera, artifacts, distortion, low resolution, overexposed, underexposed, color banding, missing details, unrealistic lighting, flickering shadows, frame stutter, ghosting, bad reflections, unrealistic motion, pixelated textures, wrong physics, broken animation, rendering artifacts, compression noise, jitter, unnatural sand behavior, visual glitches"&&Z(He.negative_prompt)}}catch{}},Is=()=>{T(null),C(""),St(null),I.current&&(I.current.value="")},Ps=async()=>{if(!S){R("Log in om video's te genereren");return}if(!P){Oe("Image is required");return}ke(!0),Oe("");const H=h*X,be=new FormData;if(be.append("file",P),be.append("num_frames",String(H)),be.append("resolution",te),be.append("fps",String(X)),be.append("aspect_ratio",W),!fe){const Yt=bu(u)+(U||"Motion, subject moving naturally");be.append("prompt",Yt)}let Te,Le=!0;fe?(Te=`${ve}/generate-pose`,Le=!1):(Te=`${ve}/generate-wan22-async`,be.append("steps",String(m)),be.append("cfg",String(q)),be.append("seed",String(F)),zn&&Mt>1&&(be.append("extend_mode","true"),be.append("clip_count",String(Mt))),_t&&be.append("unet_high_noise",_t),kt&&be.append("unet_low_noise",kt),Ne.length>0&&be.append("lora_configs",JSON.stringify(Ne)));try{const He=await ht(Te,be);if(!He.ok){Oe(He.data?.detail||`Generation failed (status ${He.status})`);return}if(Le)p&&p(He.data);else{const Yt=He.data?.video_url||He.data?.url,Jn=He.data?.output_video,Zn=Yt?`${ve}${Yt}`:"";c({kind:"video",url:Zn,backendUrl:Zn,filename:Jn,meta:He.data}),x()}}catch(He){const Yt=He?.message||"Failed to generate video";Oe(Yt),await Wa({level:"error",message:"Image-to-video failed",timestamp:new Date().toISOString(),meta:{message:Yt,modelMode:xe}})}finally{ke(!1)}};return t.jsxs("div",{className:"tool-container",children:[t.jsx("style",{children:`
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}),t.jsxs("div",{className:"grok-card",children:[t.jsx("div",{className:"grok-card-header",children:t.jsx("div",{className:"grok-card-title",children:"Model Selection"})}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Generation Mode"}),t.jsxs("div",{style:{position:"relative"},children:[t.jsx("select",{value:xe,onChange:H=>{ge(H.target.value),H.target.value==="wan2.2"&&(re("576p"),G("9:16"),v(6))},style:{width:"100%",padding:"12px 40px 12px 16px",backgroundColor:"var(--bg-secondary, #1a1a1a)",border:"1px solid var(--border-color)",borderRadius:"8px",color:"var(--text-primary, #fff)",fontSize:"1rem",appearance:"none",cursor:"pointer"},children:dh.map(H=>t.jsx("option",{value:H.value,style:{backgroundColor:"#1a1a1a",color:"#fff"},children:H.label},H.value))}),t.jsx(Qt,{size:20,style:{position:"absolute",right:"12px",top:"50%",transform:"translateY(-50%)",pointerEvents:"none",color:"var(--text-muted)"}})]}),t.jsxs("div",{className:"info-badge",style:{marginTop:"8px"},children:[t.jsx("span",{style:{fontWeight:600},children:"🎬 Wan2.2 14B Q6"})," • ",t.jsx("span",{style:{color:"#93c5fd"},children:"ComfyUI Backend"}),t.jsx("div",{style:{marginTop:"4px",opacity:.8},children:"High-quality I2V with DisTorch2 + SageAttention (10GB VRAM)"})]})]}),t.jsxs("div",{style:{marginTop:"12px",paddingTop:"12px",borderTop:"1px solid var(--border-color)"},children:[t.jsxs("div",{onClick:()=>hr(!pn),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer",padding:"4px 0"},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[t.jsx(hu,{size:16}),t.jsx("span",{style:{fontSize:"0.9rem",fontWeight:500},children:"Unet Model"}),t.jsxs("span",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:["(",_t.replace(".gguf","").replace("wan2.2_i2v_",""),")"]})]}),t.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:pn?"▼":"▶"})]}),pn&&t.jsxs("div",{style:{marginTop:"12px",display:"flex",flexDirection:"column",gap:"12px"},children:[t.jsxs("div",{children:[t.jsx("label",{style:{display:"block",fontSize:"0.8rem",color:"var(--text-muted)",marginBottom:"4px"},children:"Model Pair (recommended)"}),t.jsx("select",{onChange:H=>{const be=Je.pairs?.find(Te=>Te.name===H.target.value);be&&(We(be.high.path),_n(be.low.path))},style:{width:"100%",padding:"8px 12px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"6px",color:"var(--text-primary)",fontSize:"0.85rem"},value:Je.pairs?.find(H=>H.high.path===_t&&H.low.path===kt)?.name||"",children:Je.pairs?.map(H=>t.jsxs("option",{value:H.name,children:[H.name," (",H.high.size_gb,"GB)"]},H.name))})]}),t.jsxs("details",{style:{fontSize:"0.8rem"},children:[t.jsx("summary",{style:{cursor:"pointer",color:"var(--text-muted)",marginBottom:"8px"},children:"⚙️ Advanced: Select models separately"}),t.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"12px",paddingTop:"8px"},children:[t.jsxs("div",{children:[t.jsx("label",{style:{display:"block",fontSize:"0.8rem",color:"var(--text-muted)",marginBottom:"4px"},children:"High Noise Model (steps 0-3)"}),t.jsx("select",{value:_t,onChange:H=>We(H.target.value),style:{width:"100%",padding:"8px 12px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"6px",color:"var(--text-primary)",fontSize:"0.85rem"},children:Je.high_noise?.map(H=>t.jsxs("option",{value:H.path,children:[H.name," (",H.size_gb,"GB)"]},H.path))})]}),t.jsxs("div",{children:[t.jsx("label",{style:{display:"block",fontSize:"0.8rem",color:"var(--text-muted)",marginBottom:"4px"},children:"Low Noise Model (steps 3+)"}),t.jsx("select",{value:kt,onChange:H=>_n(H.target.value),style:{width:"100%",padding:"8px 12px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"6px",color:"var(--text-primary)",fontSize:"0.85rem"},children:Je.low_noise?.map(H=>t.jsxs("option",{value:H.path,children:[H.name," (",H.size_gb,"GB)"]},H.path))})]})]})]})]})]})]}),t.jsxs("div",{className:"grok-card",children:[t.jsxs("div",{className:"grok-card-header",children:[t.jsxs("div",{className:"grok-card-title",style:{display:"flex",alignItems:"center",gap:"6px"},children:["Positive Prompt ",t.jsx("span",{style:{fontWeight:400,color:"var(--text-muted)",fontSize:"0.85rem"},children:"(Describe the motion)"}),t.jsxs("div",{style:{position:"relative",display:"inline-block"},children:[t.jsx("button",{className:"icon-btn",style:{width:"20px",height:"20px",border:"none",background:"transparent",padding:0,fontSize:"14px"},onClick:()=>B(!k),title:"Prompt tips",children:k?"💡":"❓"}),k&&t.jsxs("div",{style:{position:"absolute",top:"100%",left:"50%",transform:"translateX(-50%)",marginTop:"8px",backgroundColor:"#1a1a1a",border:"1px solid #fbbf24",borderRadius:"8px",padding:"12px",width:"280px",zIndex:100,fontSize:"0.8rem",color:"#fbbf24",boxShadow:"0 4px 12px rgba(0,0,0,0.5)"},children:[t.jsx("div",{style:{fontWeight:600,marginBottom:"8px"},children:"💡 Prompt Tips"}),t.jsxs("ul",{style:{margin:0,paddingLeft:"16px",lineHeight:1.6},children:[t.jsx("li",{children:"Structure: [subject + motion] + [scene] + [camera]"}),t.jsx("li",{children:'Focus on motion - "walking slowly", "hair blowing"'}),t.jsx("li",{children:'Add intensity - "quickly", "gently", "dramatically"'}),t.jsx("li",{children:'Camera moves - "slow zoom in", "pan left"'}),t.jsx("li",{children:"Describe what you want, not what to avoid"})]})]})]})]}),t.jsxs("div",{style:{display:"flex",gap:"4px"},children:[t.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px",fontSize:"14px"},onClick:async()=>{if(A)try{const be=await(await fetch(`${ve}/extract-metadata-url`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image_url:A})})).json();be.positive_prompt&&D(be.positive_prompt),be.negative_prompt&&setNegPrompt(be.negative_prompt)}catch(H){console.error("Extract metadata failed:",H)}},title:"Extract prompt from selected image",disabled:!A,children:"🔍"}),t.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px",fontSize:"12px"},title:"Show prompt tips",children:"📝"}),t.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px",fontSize:"14px"},onClick:()=>D(ei(b)),title:"Generate random creative prompt",children:"✨"})]})]}),t.jsx(ju,{value:u,onChange:he}),t.jsxs("div",{style:{position:"relative"},children:[t.jsx("textarea",{className:"form-textarea",value:U,onChange:H=>D(H.target.value),rows:4,placeholder:"Describe how you want the image to move or animate... (Optional for image-to-video)",style:{backgroundColor:"#0f0f0f",border:"1px solid var(--border-color)",borderRadius:"8px",resize:"vertical",minHeight:"80px",padding:"12px",paddingBottom:"28px",width:"100%",boxSizing:"border-box"}}),t.jsxs("div",{style:{position:"absolute",bottom:"8px",right:"8px",fontSize:"0.7rem",color:"var(--text-muted)"},children:[U.length,"/2048"]})]}),t.jsxs("div",{style:{marginTop:"12px"},children:[t.jsxs("div",{onClick:()=>j(!K),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer",padding:"8px 0"},children:[t.jsx("span",{style:{fontSize:"0.85rem",color:"var(--text-secondary)"},children:"Negative Prompt"}),t.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:K?"▼":"▶"})]}),K&&t.jsxs("div",{style:{position:"relative"},children:[t.jsx("textarea",{className:"form-textarea",value:ee,onChange:H=>Z(H.target.value),rows:3,placeholder:"Things to avoid in the generation...",style:{backgroundColor:"#0f0f0f",border:"1px solid var(--border-color)",borderRadius:"8px",resize:"vertical",minHeight:"60px",padding:"12px",paddingBottom:"28px",width:"100%",boxSizing:"border-box",fontSize:"0.85rem"}}),t.jsxs("div",{style:{position:"absolute",bottom:"8px",right:"8px",fontSize:"0.7rem",color:"var(--text-muted)"},children:[ee.length,"/2048"]})]})]})]}),t.jsxs("div",{className:"grok-card",children:[t.jsx("div",{className:"grok-card-header",children:t.jsx("div",{className:"grok-card-title",children:"Upload Photo"})}),t.jsxs("div",{className:"grok-tabs",children:[t.jsxs("button",{className:`grok-tab ${L==="file"?"active":""}`,onClick:()=>V("file"),children:[t.jsx(pt,{size:14})," Upload File"]}),t.jsxs("button",{className:`grok-tab ${L==="url"?"active":""}`,onClick:()=>V("url"),children:[t.jsx(mu,{size:14})," From URL"]}),t.jsxs("button",{className:`grok-tab ${L==="creations"?"active":""}`,onClick:()=>V("creations"),children:[t.jsx(mm,{size:14})," From My Creations"]})]}),t.jsx("input",{ref:I,type:"file",accept:"image/*",onChange:H=>Hr(H.target.files?.[0]),style:{display:"none"}}),L==="file"&&!P&&t.jsxs("div",{className:"upload-box",onClick:()=>I.current?.click(),style:{cursor:"pointer",borderStyle:"dashed",minHeight:"200px",justifyContent:"center"},children:[t.jsx(pt,{size:48,className:"text-muted",style:{opacity:.2}}),t.jsx("div",{style:{fontSize:"1rem",fontWeight:500,color:"var(--text-secondary)"},children:"Drag & drop an image here, or click to browse"}),t.jsx("div",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:"JPEG, PNG, WebP, Max 20MB"}),t.jsx("div",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:"Minimum size: 300x300px"})]}),L==="url"&&!P&&t.jsxs("div",{style:{padding:"16px 0"},children:[t.jsx("div",{style:{fontSize:"0.85rem",color:"var(--text-muted)",marginBottom:"8px"},children:"Enter image URL:"}),t.jsx("input",{type:"url",placeholder:"https://example.com/image.jpg",style:{width:"100%",padding:"12px",background:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"8px",color:"var(--text-primary)",fontSize:"0.9rem"},onKeyDown:async H=>{if(H.key==="Enter"&&H.target.value)try{const Te=await(await fetch(H.target.value)).blob(),Le=H.target.value.split("/").pop()||"image.jpg",He=new File([Te],Le,{type:Te.type});Hr(He)}catch{Oe("Failed to load image from URL")}}}),t.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"8px"},children:"Press Enter to load"})]}),L==="creations"&&!P&&t.jsxs("div",{style:{padding:"24px 16px",textAlign:"center",color:"var(--text-muted)",backgroundColor:"var(--bg-secondary)",borderRadius:"8px",border:"1px dashed var(--border-color)"},children:[t.jsx(Nn,{size:32,style:{opacity:.5,marginBottom:"12px"}}),t.jsx("div",{style:{fontSize:"0.9rem",marginBottom:"8px"},children:"Select an image from the panel on the right →"}),t.jsx("div",{style:{fontSize:"0.8rem",opacity:.7},children:"Browse your generated images"})]}),P&&t.jsxs("div",{className:"relative",style:{position:"relative"},children:[t.jsx("img",{src:A,alt:"Preview",style:{width:"100%",maxHeight:"400px",objectFit:"contain",borderRadius:"8px",border:"1px solid var(--border-color)"}}),t.jsx("button",{onClick:H=>{H.stopPropagation(),Is()},style:{position:"absolute",top:"12px",right:"12px",background:"rgba(0,0,0,0.7)",border:"none",color:"white",borderRadius:"50%",width:"32px",height:"32px",display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",backdropFilter:"blur(4px)"},children:t.jsx(lt,{size:18})})]})]}),t.jsxs("div",{className:"grok-card",children:[t.jsxs("div",{className:"form-group",children:[t.jsxs("label",{className:"grok-section-label",children:["Resolution",t.jsx("span",{className:"text-muted",style:{fontWeight:400},children:" (Higher = Better Quality, more VRAM)"})]}),t.jsx("div",{className:"grok-toggle-group",children:Object.entries(uh).map(([H,be])=>t.jsxs("button",{className:`grok-toggle-btn ${te===H?"active":""}`,onClick:()=>re(H),children:[be.label,t.jsx("span",{style:{fontSize:"0.7rem",opacity:.7,display:"block"},children:be.dimensions[W]||be.dimensions["1:1"]})]},H))})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Aspect Ratio"}),t.jsx("div",{className:"grok-toggle-group",children:ph.map(H=>t.jsx("button",{className:`grok-toggle-btn ${W===H?"active":""}`,onClick:()=>G(H),children:H},H))})]}),t.jsxs("div",{className:"form-group",children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"8px"},children:[t.jsx("label",{className:"grok-section-label",children:"Duration"}),t.jsxs("span",{className:"nav-badge",style:{fontSize:"0.8rem"},children:[h,"s (",h*X,"f)"]})]}),t.jsxs("div",{style:{position:"relative",height:"24px",marginBottom:"8px"},children:[t.jsx("input",{type:"range",min:"3",max:"15",step:"1",value:h,onChange:H=>v(parseInt(H.target.value,10)),style:{width:"100%",opacity:0,position:"absolute",zIndex:2,cursor:"pointer"}}),t.jsx("div",{style:{position:"absolute",top:"10px",left:0,right:0,height:"4px",backgroundColor:"#333",borderRadius:"2px"},children:t.jsx("div",{style:{width:`${(h-3)/12*100}%`,height:"100%",backgroundColor:"var(--accent-color, #a855f7)",borderRadius:"2px"}})}),t.jsx("div",{style:{position:"absolute",top:"2px",left:`calc(${(h-3)/12*100}% - 10px)`,width:"20px",height:"20px",backgroundColor:"white",borderRadius:"50%",boxShadow:"0 2px 4px rgba(0,0,0,0.3)"}})]}),t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.75rem",color:"var(--text-muted)"},children:[t.jsx("span",{children:"3s"}),t.jsx("span",{children:"6s (rec)"}),t.jsx("span",{children:"15s"})]})]}),t.jsxs("div",{className:"form-group",children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"8px"},children:[t.jsx("label",{className:"grok-section-label",children:"Frame Rate (FPS)"}),t.jsxs("span",{className:"nav-badge",style:{fontSize:"0.8rem"},children:[X," fps"]})]}),t.jsx("div",{className:"grok-toggle-group",children:ch.map(H=>t.jsx("button",{className:`grok-toggle-btn ${X===H?"active":""}`,onClick:()=>J(H),type:"button",children:H},H))}),t.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"8px"},children:"Higher FPS = smoother motion, more VRAM required"})]}),xe!=="wan2.2"&&t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Model Version"}),t.jsxs("div",{className:"grok-toggle-group",children:[t.jsx("button",{className:`grok-toggle-btn ${E==="v1"?"active":""}`,onClick:()=>ue("v1"),children:"V1"}),t.jsx("button",{className:`grok-toggle-btn ${E==="v2"?"active":""}`,onClick:()=>ue("v2"),children:"V2 (Enhanced)"})]}),t.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"8px"},children:"V2 features improved video quality, motion, and optional audio generation"})]}),xe==="wan2.2"&&t.jsxs("div",{style:{backgroundColor:"var(--bg-tertiary)",padding:"16px",borderRadius:"8px",marginTop:"8px"},children:[t.jsxs("div",{onClick:()=>$t(!mn),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer"},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[t.jsx(zs,{size:16}),t.jsx("span",{style:{fontWeight:600,fontSize:"0.9rem"},children:"Workflow Presets"}),Fe&&t.jsx("span",{style:{fontSize:"0.7rem",backgroundColor:"var(--accent-color)",color:"white",padding:"2px 6px",borderRadius:"4px",marginLeft:"4px"},children:Fe.name})]}),t.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:mn?"▼":"▶"})]}),mn&&t.jsx("div",{style:{marginTop:"12px"},children:t.jsx(lh,{onPresetChange:H=>{if(Wr(H),H?.parameters){const be=H.parameters;be.steps?.default&&$(be.steps.default),be.cfg?.default&&le(be.cfg.default),be.seed?.default!==void 0&&_(be.seed.default),be.frame_rate?.default&&J(be.frame_rate.default)}},onParametersChange:H=>{ce(H),H.steps!==void 0&&$(H.steps),H.cfg!==void 0&&le(H.cfg),H.seed!==void 0&&_(H.seed),H.frame_rate!==void 0&&J(H.frame_rate)},currentParameters:{steps:m,cfg:q,seed:F,frame_rate:X}})})]}),xe==="wan2.2"&&t.jsxs("div",{style:{backgroundColor:"var(--bg-tertiary)",padding:"16px",borderRadius:"8px",marginTop:"8px"},children:[t.jsxs("div",{onClick:()=>Q(!Y),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer"},children:[t.jsx("div",{style:{fontSize:"0.9rem",fontWeight:600,color:"var(--text-primary)"},children:"⚙️ Sampling Settings"}),t.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:Y?"▼":"▶"})]}),Y&&t.jsxs("div",{style:{marginTop:"12px"},children:[t.jsxs("div",{className:"form-group",style:{marginBottom:"12px"},children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[t.jsx("label",{className:"grok-section-label",children:"Sampling Steps"}),t.jsx("span",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:m})]}),t.jsx("input",{type:"range",min:"4",max:"20",step:"1",value:m,onChange:H=>$(parseInt(H.target.value,10)),style:{width:"100%",cursor:"pointer"}}),t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.7rem",color:"var(--text-muted)"},children:[t.jsx("span",{children:"4 (fast)"}),t.jsx("span",{children:"6 (rec)"}),t.jsx("span",{children:"20 (quality)"})]})]}),t.jsxs("div",{className:"form-group",style:{marginBottom:"12px"},children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[t.jsx("label",{className:"grok-section-label",children:"CFG Guidance"}),t.jsx("span",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:q.toFixed(1)})]}),t.jsx("input",{type:"range",min:"1.0",max:"10.0",step:"0.5",value:q,onChange:H=>le(parseFloat(H.target.value)),style:{width:"100%",cursor:"pointer"}}),t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.7rem",color:"var(--text-muted)"},children:[t.jsx("span",{children:"1.0 (rec)"}),t.jsx("span",{children:"5.0"}),t.jsx("span",{children:"10.0"})]})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Seed"}),t.jsxs("div",{style:{display:"flex",gap:"8px"},children:[t.jsx("input",{type:"number",value:F,onChange:H=>_(parseInt(H.target.value,10)),placeholder:"-1 for random",style:{flex:1,padding:"8px 12px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"6px",color:"var(--text-primary)",fontSize:"0.9rem"}}),t.jsx("button",{className:"btn ghost sm",onClick:()=>_(-1),style:{whiteSpace:"nowrap"},children:"Random"})]}),t.jsx("div",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginTop:"4px"},children:"-1 = random seed each generation"})]})]}),t.jsxs("div",{style:{marginTop:"16px",paddingTop:"16px",borderTop:"1px solid var(--border-color)"},children:[t.jsxs("div",{onClick:()=>it(!Ve),style:{display:"flex",justifyContent:"space-between",alignItems:"center",cursor:"pointer",marginBottom:Ve?"12px":0},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[t.jsx(fu,{size:16}),t.jsx("span",{style:{fontWeight:600,fontSize:"0.9rem"},children:"LoRA Models"}),Ne.length>0&&t.jsxs("span",{style:{fontSize:"0.7rem",backgroundColor:"var(--accent-color)",color:"white",padding:"2px 6px",borderRadius:"4px"},children:[Ne.length," active"]})]}),t.jsx("span",{style:{opacity:.5,fontSize:"0.8rem"},children:Ve?"▼":"▶"})]}),Ve&&t.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"12px"},children:[Ne.map((H,be)=>t.jsxs("div",{style:{backgroundColor:"var(--bg-input)",borderRadius:"8px",padding:"12px",border:"1px solid var(--border-color)"},children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"8px"},children:[t.jsxs("span",{style:{fontSize:"0.8rem",fontWeight:600},children:["LoRA #",be+1]}),t.jsx("button",{onClick:()=>Re(Ne.filter((Te,Le)=>Le!==be)),style:{background:"transparent",border:"none",color:"#ef4444",cursor:"pointer",padding:"2px 6px",fontSize:"0.8rem"},children:"✕ Remove"})]}),t.jsxs("div",{style:{marginBottom:"8px"},children:[t.jsx("label",{style:{display:"block",fontSize:"0.75rem",color:"var(--text-muted)",marginBottom:"4px"},children:"High Noise (steps 0-3)"}),t.jsxs("select",{value:H.high||"",onChange:Te=>{const Le=[...Ne];Le[be]={...H,high:Te.target.value},Re(Le)},style:{width:"100%",padding:"6px 10px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"4px",color:"var(--text-primary)",fontSize:"0.8rem"},children:[t.jsx("option",{value:"",children:"None"}),Lt.by_category&&Object.keys(Lt.by_category).sort().map(Te=>t.jsx("optgroup",{label:Te==="root"?"📁 Root":`📁 ${Te}`,children:Lt.by_category[Te].map(Le=>t.jsxs("option",{value:Le.path,children:[Le.name," (",Le.size_mb,"MB)"]},Le.path))},Te))]})]}),t.jsxs("div",{style:{marginBottom:"8px"},children:[t.jsx("label",{style:{display:"block",fontSize:"0.75rem",color:"var(--text-muted)",marginBottom:"4px"},children:"Low Noise (steps 3+)"}),t.jsxs("select",{value:H.low||"",onChange:Te=>{const Le=[...Ne];Le[be]={...H,low:Te.target.value},Re(Le)},style:{width:"100%",padding:"6px 10px",backgroundColor:"var(--bg-secondary)",border:"1px solid var(--border-color)",borderRadius:"4px",color:"var(--text-primary)",fontSize:"0.8rem"},children:[t.jsx("option",{value:"",children:"None (uses High Noise)"}),Lt.by_category&&Object.keys(Lt.by_category).sort().map(Te=>t.jsx("optgroup",{label:Te==="root"?"📁 Root":`📁 ${Te}`,children:Lt.by_category[Te].map(Le=>t.jsxs("option",{value:Le.path,children:[Le.name," (",Le.size_mb,"MB)"]},Le.path))},Te))]})]}),t.jsxs("div",{children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"2px"},children:[t.jsx("label",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Strength"}),t.jsx("span",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:(H.strength||1).toFixed(2)})]}),t.jsx("input",{type:"range",min:"0",max:"2",step:"0.05",value:H.strength||1,onChange:Te=>{const Le=[...Ne];Le[be]={...H,strength:parseFloat(Te.target.value)},Re(Le)},style:{width:"100%",cursor:"pointer"}})]})]},be)),t.jsx("button",{onClick:()=>Re([...Ne,{high:"",low:"",strength:1}]),style:{padding:"8px 12px",backgroundColor:"transparent",border:"1px dashed var(--border-color)",borderRadius:"6px",color:"var(--text-secondary)",cursor:"pointer",fontSize:"0.85rem",display:"flex",alignItems:"center",justifyContent:"center",gap:"6px"},children:"+ Add LoRA"}),t.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",fontStyle:"italic"},children:"💡 Stack multiple LoRAs for combined effects. Each LoRA has its own strength."})]})]})]}),t.jsxs("div",{className:"form-group",style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[t.jsxs("div",{children:[t.jsx("div",{className:"grok-section-label",style:{marginBottom:"4px"},children:"Generate Audio"}),t.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Enable audio generation (increases credits)"})]}),t.jsxs("label",{className:"grok-switch",children:[t.jsx("input",{type:"checkbox"}),t.jsx("span",{className:"grok-slider"})]})]}),t.jsxs("div",{className:"form-group",style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[t.jsxs("div",{children:[t.jsx("div",{className:"grok-section-label",style:{marginBottom:"4px"},children:"Camera Fixed"}),t.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Whether to fix the camera position"})]}),t.jsxs("label",{className:"grok-switch",children:[t.jsx("input",{type:"checkbox"}),t.jsx("span",{className:"grok-slider"})]})]}),t.jsxs("div",{className:"form-group",style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[t.jsxs("div",{children:[t.jsx("div",{className:"grok-section-label",style:{marginBottom:"4px"},children:"🎬 Extend Duration"}),t.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:"Chain multiple clips sequentially"})]}),t.jsxs("label",{className:"grok-switch",children:[t.jsx("input",{type:"checkbox",checked:zn,onChange:H=>{gr(H.target.checked),H.target.checked||fn(1)}}),t.jsx("span",{className:"grok-slider"})]})]}),zn&&t.jsxs("div",{className:"form-group",style:{background:"linear-gradient(135deg, rgba(233, 69, 96, 0.1) 0%, rgba(233, 69, 96, 0.05) 100%)",borderRadius:"8px",padding:"12px",marginTop:"-8px",border:"1px solid rgba(233, 69, 96, 0.2)"},children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"8px"},children:[t.jsxs("div",{className:"grok-section-label",children:["Number of Clips: ",Mt]}),t.jsxs("div",{style:{fontSize:"0.75rem",color:"#e94560",background:"rgba(233, 69, 96, 0.15)",padding:"2px 8px",borderRadius:"10px",fontWeight:"600"},children:["≈ ",(h*Mt).toFixed(0),"s total"]})]}),t.jsx("input",{type:"range",min:"1",max:"5",value:Mt,onChange:H=>fn(parseInt(H.target.value)),style:{width:"100%",accentColor:"#e94560"}}),t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.7rem",color:"var(--text-muted)",marginTop:"4px"},children:[t.jsx("span",{children:"1"}),t.jsx("span",{children:"2"}),t.jsx("span",{children:"3"}),t.jsx("span",{children:"4"}),t.jsx("span",{children:"5"})]}),t.jsx("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"8px",fontStyle:"italic"},children:"🔗 Each clip continues from the last frame of the previous clip"})]})]}),t.jsxs("div",{className:"grok-card",children:[t.jsx("div",{className:"grok-card-header",children:t.jsx("div",{className:"grok-card-title",children:"Aspect Ratio"})}),t.jsx("div",{className:"aspect-grid",children:[{label:"Auto",icon:t.jsx(Cn,{size:16})},{label:"21:9",icon:t.jsx("div",{style:{width:"24px",height:"10px",border:"1px solid currentColor"}})},{label:"16:9",icon:t.jsx("div",{style:{width:"24px",height:"14px",border:"1px solid currentColor"}})},{label:"4:3",icon:t.jsx("div",{style:{width:"20px",height:"15px",border:"1px solid currentColor"}})},{label:"1:1",icon:t.jsx("div",{style:{width:"18px",height:"18px",border:"1px solid currentColor"}})},{label:"3:4",icon:t.jsx("div",{style:{width:"15px",height:"20px",border:"1px solid currentColor"}})},{label:"9:16",icon:t.jsx("div",{style:{width:"14px",height:"24px",border:"1px solid currentColor"}})}].map(H=>t.jsxs("button",{className:`aspect-btn ${W===H.label?"active":""}`,onClick:()=>G(H.label),children:[t.jsx("div",{className:"aspect-icon",style:{background:"transparent",display:"flex",alignItems:"center",justifyContent:"center",border:"none"},children:H.icon}),t.jsx("span",{className:"aspect-label",children:H.label})]},H.label))})]}),Ae&&t.jsx("div",{style:{padding:"12px",backgroundColor:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.2)",borderRadius:"8px",color:"#ef4444",marginBottom:"16px",fontSize:"0.9rem"},children:Ae}),!ae&&Qe&&t.jsxs("div",{style:{display:"flex",alignItems:"center",justifyContent:"center",gap:"6px",marginBottom:"8px",fontSize:"0.85rem",color:"var(--text-muted)"},children:[t.jsx(Qn,{size:14}),t.jsxs("span",{children:["Estimated time: ~",Ge.formatted]})]}),t.jsx("button",{className:"primary-btn",disabled:!Qe,onClick:Ps,style:{height:"48px",fontSize:"1rem",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px",backgroundColor:"#e5e5e5",color:"black"},children:ae?t.jsx(t.Fragment,{children:"Generating..."}):t.jsxs(t.Fragment,{children:[t.jsx(Cn,{size:18}),"Generate from Image"]})}),ae&&t.jsx("div",{className:"progress-container",children:t.jsx("div",{className:"progress-indeterminate"})})]})}const dn={wan22:[{value:"wan2.2-t2i",label:"Wan2.2 T2I (Multi-GPU)",category:"Video Model"}],flux:[{value:"flux1-dev-fp8",label:"Flux.1 Dev (FP8)",category:"Flux"}],sdxl:[{value:"CyberRealistic_Pony_v14.1_FP16.safetensors",label:"CyberRealistic Pony",category:"Realistic/Pony"},{value:"dreamshaperXL_lightningDPMSDE.safetensors",label:"Dreamshaper Lightning",category:"General"},{value:"illustriousRealismBy_v10VAE.safetensors",label:"Illustrious Realism",category:"Realistic"},{value:"juggernautXL_ragnarok.safetensors",label:"Juggernaut XL",category:"General"},{value:"novaAnimeXL_ilV150.safetensors",label:"Nova Anime XL",category:"Anime"},{value:"ponyDiffusionV6XL_v6StartWithThisOne.safetensors",label:"Pony Diffusion V6",category:"Pony"},{value:"reapony_v90.safetensors",label:"Reapony V9",category:"Realistic/Pony"},{value:"ultraRealisticByStable_v20FP16.safetensors",label:"Ultra Realistic",category:"Realistic"},{value:"waiIllustriousSDXL_v160.safetensors",label:"Wai Illustrious",category:"Anime"}],sd15:[{value:"Realistic_Vision_V5.1.safetensors",label:"Realistic Vision V5.1",category:"Realistic"}],diffusers:[{value:"sd3.5-large-int8",label:"SD3.5 Large (INT8)"}]},Sn=c=>c==="wan2.2-t2i"?"wan22":c.startsWith("flux")?"flux":c==="Realistic_Vision_V5.1.safetensors"?"sd15":c.endsWith(".safetensors")?"sdxl":"diffusers";function mh({onOutput:c,onJobSubmitted:x}){const{nsfwEnabled:d}=Ga(),{user:N,requestLogin:p}=Ke(),[b,S]=l.useState(""),[R,I]=l.useState("ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text"),[P,T]=l.useState("1:1"),[A,C]=l.useState("normal"),[L,V]=l.useState("CyberRealistic_Pony_v14.1_FP16.safetensors"),[U,D]=l.useState(1),[ee,Z]=l.useState(!1),[K,j]=l.useState(""),[k,B]=l.useState(!1),[h,v]=l.useState(null),[te,re]=l.useState([]),[xe,ge]=l.useState([{name:"None",strength:1},{name:"None",strength:1},{name:"None",strength:1}]),[E,ue]=l.useState(30),[fe,ie]=l.useState(7.5),[W,G]=l.useState(3.5),[X,J]=l.useState(-1),[m,$]=l.useState("dpmpp_2m"),[q,le]=l.useState("karras");l.useEffect(()=>{(async()=>{try{const he=await fetch(`${ve}/loras`);if(he.ok){const ze=await he.json();re(ze.loras||[])}}catch(he){console.warn("Failed to fetch LoRAs:",he)}})()},[]);const F=l.useMemo(()=>d?te:te.filter(u=>!u.nsfw),[te,d]),_=(u,he,ze)=>{ge(pe=>{const Ne=[...pe];return Ne[u]={...Ne[u],[he]:ze},Ne})},Y=async()=>{if(!N){p("Log in om te genereren");return}if(b.trim()){Z(!0),j(""),v(null);try{const u=[];for(let he=0;he<U;he++){const ze=`t2i-${Date.now()}-${Math.random().toString(36).slice(2,8)}`,pe=new FormData;pe.append("prompt",b),pe.append("aspect_ratio",P);const Ne=Sn(L);let Re="/generate-image";if(Ne==="wan22")Re="/generate-wan22-t2i",pe.append("steps",E),pe.append("seed",X);else if(Ne==="flux")Re="/generate-flux",pe.append("steps",E),pe.append("guidance",W),pe.append("seed",X);else if(Ne==="sdxl"){Re="/generate-sdxl",pe.append("checkpoint",L),pe.append("negative_prompt",R),pe.append("steps",E),pe.append("cfg",fe),pe.append("seed",X),pe.append("sampler_name",m),pe.append("scheduler",q);const it=xe.filter(Je=>Je.name&&Je.name!=="None");it.length>0&&pe.append("lora_configs",JSON.stringify(it))}else Ne==="sd15"?(Re="/generate-sd15",pe.append("negative_prompt",R),pe.append("steps",E),pe.append("cfg",fe),pe.append("seed",X),pe.append("sampler_name",m),pe.append("scheduler",q)):(pe.append("mode",A),pe.append("model",L),pe.append("job_id",ze));const Ve=await ht(`${ve}${Re}`,pe);if(!Ve.ok)throw new Error(Ve.data?.detail||`Generation failed (status ${Ve.status})`);Ve.data?.prompt_id&&u.push(Ve.data.prompt_id),x&&x({prompt_id:Ve.data?.prompt_id})}v({count:U,model:Q(),promptIds:u})}catch(u){console.error("Generation error:",u),j(u.message||"Failed to generate image")}finally{Z(!1)}}},Q=()=>[...dn.wan22,...dn.flux,...dn.sdxl,...dn.sd15,...dn.diffusers].find(ze=>ze.value===L)?.label||L;return t.jsxs("div",{className:"tool-container",children:[t.jsx("div",{className:"grok-card",children:t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Mode"}),t.jsxs("div",{className:"form-select",style:{display:"flex",alignItems:"center",gap:"8px",cursor:"pointer"},children:[t.jsx(Cn,{size:16,className:"text-primary"}),t.jsx("span",{children:"Normal"})]}),t.jsxs("div",{className:"info-badge",children:[t.jsx("span",{style:{color:"#93c5fd"},children:"Standard Quality"}),t.jsx("div",{style:{marginTop:"4px",opacity:.8},children:"Fast and efficient image generation (1 credit per image)"})]})]})}),t.jsxs("div",{className:"grok-card",children:[t.jsxs("div",{className:"grok-card-header",children:[t.jsx("div",{className:"grok-card-title",children:"Enter Image Prompt"}),t.jsxs("div",{style:{display:"flex",gap:"4px"},children:[t.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px",fontSize:"10px"},children:"T"}),t.jsx("button",{className:"icon-btn",style:{width:"24px",height:"24px",fontSize:"10px"},children:"✨"})]})]}),t.jsx("div",{style:{position:"relative"},children:t.jsx("textarea",{className:"form-textarea",value:b,onChange:u=>S(u.target.value),rows:4,placeholder:"A attractive blonde woman with cup f, tattoos, looking at me defiantly.",style:{backgroundColor:"#0f0f0f",border:"none",resize:"none",paddingBottom:"24px"}})})]}),t.jsxs("div",{className:"grok-card",children:[t.jsxs("div",{className:"grok-card-header",children:[t.jsx("div",{className:"grok-card-title",children:"Model"}),t.jsx("span",{className:"nav-badge",style:{fontSize:"0.7rem"},children:Sn(L).toUpperCase()})]}),t.jsxs("div",{style:{marginBottom:"12px"},children:[t.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"⚡ Flux (Best Quality)"}),t.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:dn.flux.map(u=>t.jsx("button",{className:`grok-toggle-btn ${L===u.value?"active":""}`,onClick:()=>V(u.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},children:u.label},u.value))})]}),t.jsxs("div",{style:{marginBottom:"12px"},children:[t.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"🎨 SDXL Checkpoints"}),t.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:dn.sdxl.map(u=>t.jsx("button",{className:`grok-toggle-btn ${L===u.value?"active":""}`,onClick:()=>V(u.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},title:u.category,children:u.label},u.value))})]}),t.jsxs("div",{style:{marginBottom:"12px"},children:[t.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"🖼️ SD 1.5 (Fast, Low VRAM)"}),t.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:dn.sd15.map(u=>t.jsx("button",{className:`grok-toggle-btn ${L===u.value?"active":""}`,onClick:()=>V(u.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},children:u.label},u.value))})]}),t.jsxs("div",{style:{marginBottom:"12px"},children:[t.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"🎬 Wan2.2 (Video Model T2I)"}),t.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:dn.wan22.map(u=>t.jsx("button",{className:`grok-toggle-btn ${L===u.value?"active":""}`,onClick:()=>V(u.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},children:u.label},u.value))})]}),t.jsxs("div",{children:[t.jsx("label",{className:"grok-section-label",style:{fontSize:"0.75rem",opacity:.7,marginBottom:"8px"},children:"🐍 Diffusers (Python)"}),t.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"6px"},children:dn.diffusers.map(u=>t.jsx("button",{className:`grok-toggle-btn ${L===u.value?"active":""}`,onClick:()=>V(u.value),style:{fontSize:"0.75rem",padding:"6px 10px",minWidth:"auto"},children:u.label},u.value))})]})]}),(Sn(L)==="sdxl"||Sn(L)==="sd15")&&t.jsxs("div",{className:"grok-card",children:[t.jsx("div",{className:"grok-card-header",children:t.jsx("div",{className:"grok-card-title",children:"Negative Prompt"})}),t.jsx("textarea",{className:"form-textarea",value:R,onChange:u=>I(u.target.value),rows:2,placeholder:"ugly, deformed, blurry...",style:{backgroundColor:"#0f0f0f",border:"none",resize:"none",fontSize:"0.85rem"}})]}),t.jsxs("div",{className:"grok-card",children:[t.jsx("div",{className:"grok-card-header",children:t.jsx("div",{className:"grok-card-title",children:"Aspect Ratio"})}),t.jsx("div",{className:"aspect-grid",style:{gridTemplateColumns:"repeat(5, 1fr)"},children:[{label:"1:1",icon:t.jsx("div",{style:{width:"18px",height:"18px",border:"1px solid currentColor"}})},{label:"16:9",icon:t.jsx("div",{style:{width:"24px",height:"14px",border:"1px solid currentColor"}})},{label:"9:16",icon:t.jsx("div",{style:{width:"14px",height:"24px",border:"1px solid currentColor"}})},{label:"4:3",icon:t.jsx("div",{style:{width:"20px",height:"15px",border:"1px solid currentColor"}})},{label:"3:4",icon:t.jsx("div",{style:{width:"15px",height:"20px",border:"1px solid currentColor"}})},{label:"2:3",icon:t.jsx("div",{style:{width:"16px",height:"24px",border:"1px solid currentColor"}})},{label:"3:2",icon:t.jsx("div",{style:{width:"24px",height:"16px",border:"1px solid currentColor"}})},{label:"4:5",icon:t.jsx("div",{style:{width:"16px",height:"20px",border:"1px solid currentColor"}})},{label:"5:4",icon:t.jsx("div",{style:{width:"20px",height:"16px",border:"1px solid currentColor"}})},{label:"9:21",icon:t.jsx("div",{style:{width:"10px",height:"24px",border:"1px solid currentColor"}})},{label:"21:9",icon:t.jsx("div",{style:{width:"24px",height:"10px",border:"1px solid currentColor"}})}].map(u=>t.jsxs("button",{className:`aspect-btn ${P===u.label?"active":""}`,onClick:()=>T(u.label),style:{height:"60px"},children:[t.jsx("div",{className:"aspect-icon",style:{background:"transparent",display:"flex",alignItems:"center",justifyContent:"center",border:"none",marginBottom:"4px"},children:u.icon}),t.jsx("span",{className:"aspect-label",style:{fontSize:"0.65rem"},children:u.label})]},u.label))})]}),t.jsxs("div",{className:"grok-card",children:[t.jsxs("div",{className:"grok-card-header",style:{cursor:"pointer"},onClick:()=>B(!k),children:[t.jsx("div",{className:"grok-card-title",children:"Advanced Settings"}),t.jsx(Qt,{size:16,className:"text-muted",style:{transform:k?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.2s"}})]}),k&&t.jsxs(t.Fragment,{children:[t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Batch Count"}),t.jsx("div",{className:"grok-toggle-group",children:[1,2,3,4].map(u=>t.jsx("button",{className:`grok-toggle-btn ${U===u?"active":""}`,onClick:()=>D(u),children:u},u))})]}),Sn(L)==="flux"&&t.jsxs(t.Fragment,{children:[t.jsxs("div",{className:"form-group",style:{marginTop:"12px"},children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[t.jsx("label",{className:"grok-section-label",children:"Steps"}),t.jsx("span",{className:"nav-badge",children:E})]}),t.jsx("input",{type:"range",min:"10",max:"30",value:E,onChange:u=>ue(parseInt(u.target.value)),className:"form-range"})]}),t.jsxs("div",{className:"form-group",children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[t.jsx("label",{className:"grok-section-label",children:"Guidance"}),t.jsx("span",{className:"nav-badge",children:W})]}),t.jsx("input",{type:"range",min:"1",max:"10",step:"0.5",value:W,onChange:u=>G(parseFloat(u.target.value)),className:"form-range"})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Seed (-1 = random)"}),t.jsx("input",{type:"number",value:X,onChange:u=>J(parseInt(u.target.value)||-1),className:"form-input",style:{backgroundColor:"#0f0f0f",border:"1px solid #333",borderRadius:"6px",padding:"8px",width:"100%"}})]})]}),Sn(L)==="wan22"&&t.jsxs(t.Fragment,{children:[t.jsxs("div",{className:"form-group",style:{marginTop:"12px"},children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[t.jsx("label",{className:"grok-section-label",children:"Steps"}),t.jsx("span",{className:"nav-badge",children:E})]}),t.jsx("input",{type:"range",min:"10",max:"50",value:E,onChange:u=>ue(parseInt(u.target.value)),className:"form-range"}),t.jsx("div",{style:{fontSize:"0.7rem",opacity:.6,marginTop:"4px"},children:"Multi-GPU workflow (DisTorch2) - 2-stage denoising"})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Seed (-1 = random)"}),t.jsx("input",{type:"number",value:X,onChange:u=>J(parseInt(u.target.value)||-1),className:"form-input",style:{backgroundColor:"#0f0f0f",border:"1px solid #333",borderRadius:"6px",padding:"8px",width:"100%"}})]})]}),(Sn(L)==="sdxl"||Sn(L)==="sd15")&&t.jsxs(t.Fragment,{children:[t.jsxs("div",{className:"form-group",style:{marginTop:"12px"},children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[t.jsx("label",{className:"grok-section-label",children:"Steps"}),t.jsx("span",{className:"nav-badge",children:E})]}),t.jsx("input",{type:"range",min:"10",max:"50",value:E,onChange:u=>ue(parseInt(u.target.value)),className:"form-range"})]}),t.jsxs("div",{className:"form-group",children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",marginBottom:"4px"},children:[t.jsx("label",{className:"grok-section-label",children:"CFG Scale"}),t.jsx("span",{className:"nav-badge",children:fe})]}),t.jsx("input",{type:"range",min:"1",max:"15",step:"0.5",value:fe,onChange:u=>ie(parseFloat(u.target.value)),className:"form-range"})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Sampler"}),t.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"4px"},children:["euler","euler_ancestral","dpmpp_2m","dpmpp_sde"].map(u=>t.jsx("button",{className:`grok-toggle-btn ${m===u?"active":""}`,onClick:()=>$(u),style:{fontSize:"0.7rem",padding:"4px 8px"},children:u},u))})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Scheduler"}),t.jsx("div",{className:"grok-toggle-group",style:{flexWrap:"wrap",gap:"4px"},children:["normal","karras","exponential","sgm_uniform"].map(u=>t.jsx("button",{className:`grok-toggle-btn ${q===u?"active":""}`,onClick:()=>le(u),style:{fontSize:"0.7rem",padding:"4px 8px"},children:u},u))})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Seed (-1 = random)"}),t.jsx("input",{type:"number",value:X,onChange:u=>J(parseInt(u.target.value)||-1),className:"form-input",style:{backgroundColor:"#0f0f0f",border:"1px solid #333",borderRadius:"6px",padding:"8px",width:"100%"}})]}),Sn(L)==="sdxl"&&F.length>0&&t.jsxs("div",{className:"form-group",children:[t.jsxs("label",{className:"grok-section-label",style:{marginBottom:"8px"},children:["LoRAs (up to 3) ",!d&&te.length>F.length&&t.jsxs("span",{style:{fontSize:"0.65rem",color:"var(--text-muted)",marginLeft:"8px"},children:["(",te.length-F.length," hidden)"]})]}),xe.map((u,he)=>t.jsxs("div",{style:{display:"flex",gap:"8px",marginBottom:"8px",alignItems:"center"},children:[t.jsxs("select",{value:u.name,onChange:ze=>_(he,"name",ze.target.value),style:{flex:1,backgroundColor:"#0f0f0f",border:"1px solid #333",borderRadius:"6px",padding:"6px 8px",color:"#fff",fontSize:"0.75rem"},children:[t.jsx("option",{value:"None",children:"None"}),F.map(ze=>t.jsx("option",{value:ze.name,children:ze.name},ze.path))]}),t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"4px",minWidth:"80px"},children:[t.jsx("input",{type:"range",min:"0",max:"2",step:"0.1",value:u.strength,onChange:ze=>_(he,"strength",parseFloat(ze.target.value)),disabled:u.name==="None",style:{width:"50px"}}),t.jsx("span",{style:{fontSize:"0.7rem",opacity:u.name==="None"?.3:1},children:u.strength.toFixed(1)})]})]},he)),t.jsx("div",{style:{fontSize:"0.65rem",opacity:.5,marginTop:"4px"},children:"Strength: 0.5-1.0 recommended"})]})]})]})]}),K&&t.jsx("div",{style:{color:"#ef4444",marginBottom:"12px",fontSize:"0.9rem"},children:K}),t.jsx("button",{className:"primary-btn",onClick:Y,disabled:ee||!b.trim(),style:{height:"48px",fontSize:"1rem",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px",backgroundColor:"white",color:"black"},children:ee?t.jsx(t.Fragment,{children:"Queueing..."}):t.jsxs(t.Fragment,{children:[t.jsx(Cn,{size:18}),"Generate ",U>1?`${U} Images`:"Image"," (",U,")"]})}),h&&t.jsxs("div",{style:{padding:"12px 16px",backgroundColor:"rgba(34, 197, 94, 0.2)",border:"1px solid rgba(34, 197, 94, 0.5)",borderRadius:"8px",color:"#86efac",fontSize:"0.875rem",marginTop:"12px"},children:["✅ ",h.count>1?`${h.count} jobs`:"Job"," queued! (",h.model,") - Check queue panel for progress"]}),K&&t.jsx("div",{style:{padding:"12px 16px",backgroundColor:"rgba(239, 68, 68, 0.2)",border:"1px solid rgba(239, 68, 68, 0.5)",borderRadius:"8px",color:"#fca5a5",fontSize:"0.875rem",marginTop:"12px"},children:K})]})}function xh({onOutput:c}){const{user:x,requestLogin:d}=Ke(),[N,p]=l.useState(""),[b,S]=l.useState("16:9"),[R,I]=l.useState(!1),[P,T]=l.useState(null),[A,C]=l.useState(""),[L,V]=l.useState(16),[U,D]=l.useState(!1),ee=async()=>{if(!x){d("Log in om te genereren");return}N.trim()&&(I(!0),setTimeout(()=>{I(!1),alert("Text-to-Image backend is not yet connected.")},1500))},Z=async()=>{P&&(D(!0),setTimeout(()=>D(!1),2e3))};return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"grok-card",children:[t.jsxs("div",{className:"grok-card-header",children:[t.jsx("div",{className:"grok-card-title",children:"Step 1: Text to Image"}),t.jsx(Nn,{size:16,className:"text-muted"})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Image Prompt"}),t.jsx("textarea",{className:"form-textarea",value:N,onChange:K=>p(K.target.value),placeholder:"Describe the image you want to generate...",rows:3,style:{backgroundColor:"#0f0f0f",border:"none",resize:"none"}})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Aspect Ratio"}),t.jsx("div",{className:"aspect-grid",children:[{label:"16:9",icon:t.jsx("div",{style:{width:"24px",height:"14px",border:"2px solid currentColor",borderRadius:"2px"}})},{label:"9:16",icon:t.jsx("div",{style:{width:"14px",height:"24px",border:"2px solid currentColor",borderRadius:"2px"}})},{label:"1:1",icon:t.jsx("div",{style:{width:"20px",height:"20px",border:"2px solid currentColor",borderRadius:"2px"}})},{label:"21:9",icon:t.jsx("div",{style:{width:"28px",height:"12px",border:"2px solid currentColor",borderRadius:"2px"}})}].map(K=>t.jsxs("button",{className:`aspect-btn ${b===K.label?"active":""}`,onClick:()=>S(K.label),children:[t.jsx("div",{className:"aspect-icon",style:{background:"transparent",display:"flex",alignItems:"center",justifyContent:"center"},children:K.icon}),t.jsx("span",{className:"aspect-label",children:K.label})]},K.label))})]}),t.jsx("button",{className:"primary-btn",onClick:ee,disabled:R||!N.trim(),style:{display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:R?"Generating Image...":t.jsxs(t.Fragment,{children:[t.jsx(Cn,{size:16})," Generate Image"]})})]}),t.jsxs("div",{className:`grok-card ${P?"":"opacity-50"}`,style:{transition:"opacity 0.3s"},children:[t.jsxs("div",{className:"grok-card-header",children:[t.jsx("div",{className:"grok-card-title",children:"Step 2: Animate"}),t.jsx(Ua,{size:16,className:"text-muted"})]}),P?t.jsx("div",{className:"form-group",children:t.jsx("img",{src:P,alt:"Generated",style:{width:"100%",borderRadius:"8px",border:"1px solid var(--border-color)",marginBottom:"16px"}})}):t.jsx("div",{className:"upload-box",style:{padding:"24px",marginBottom:"16px",borderStyle:"dashed"},children:t.jsx("div",{className:"text-muted",children:"Generate an image above to continue"})}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Motion Prompt (Optional)"}),t.jsx("textarea",{className:"form-textarea",value:A,onChange:K=>C(K.target.value),placeholder:"Describe how the image should move...",rows:2,disabled:!P,style:{backgroundColor:"#0f0f0f",border:"none",resize:"none"}})]}),t.jsxs("div",{className:"form-group",children:[t.jsxs("label",{className:"grok-section-label",children:["Duration (",L," frames)"]}),t.jsx("input",{type:"range",min:"8",max:"32",step:"4",value:L,onChange:K=>V(parseInt(K.target.value,10)),disabled:!P,style:{width:"100%",accentColor:"var(--text-primary)"}})]}),t.jsx("button",{className:"primary-btn",onClick:Z,disabled:!P||U,style:{display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:U?"Generating Video...":t.jsxs(t.Fragment,{children:[t.jsx(Ua,{size:16})," Generate Video"]})})]})]})}const hh=[{value:"none",label:"Custom",desc:"Use your own prompt"},{value:"anime",label:"Anime",desc:"Japanese animation style"},{value:"cartoon",label:"Cartoon",desc:"Cartoon/comic style"},{value:"sketch",label:"Sketch",desc:"Pencil sketch effect"},{value:"oil-painting",label:"Oil Painting",desc:"Classic oil painting style"},{value:"watercolor",label:"Watercolor",desc:"Watercolor painting effect"},{value:"pixel-art",label:"Pixel Art",desc:"Retro pixel art style"},{value:"cyberpunk",label:"Cyberpunk",desc:"Neon futuristic style"},{value:"3d-render",label:"3D Render",desc:"Modern 3D rendered look"}],gh={anime:"anime style, japanese animation, cel shading, vibrant colors, detailed linework",cartoon:"cartoon style, comic art, bold outlines, bright colors, disney style",sketch:"pencil sketch, hand-drawn, graphite, detailed linework, black and white","oil-painting":"oil painting style, classical art, brush strokes, rich colors, masterpiece",watercolor:"watercolor painting, soft edges, translucent colors, artistic, flowing","pixel-art":"pixel art style, 8-bit, retro gaming, blocky, nostalgic",cyberpunk:"cyberpunk style, neon lights, futuristic, rain, dark atmosphere, high tech","3d-render":"3d render, modern cgi, photorealistic, octane render, unreal engine"};function vh({onOutput:c,onJobSubmitted:x}){const{user:d,requestLogin:N}=Ke(),[p,b]=l.useState(null),[S,R]=l.useState(null),[I,P]=l.useState(null),[T,A]=l.useState("none"),[C,L]=l.useState(""),[V,U]=l.useState("blurry, low quality, distorted, watermark"),[D,ee]=l.useState(.5),[Z,K]=l.useState(8),[j,k]=l.useState(32),[B,h]=l.useState(!1),[v,te]=l.useState(20),[re,xe]=l.useState(7.5),[ge,E]=l.useState(-1),[ue,fe]=l.useState(!1),[ie,W]=l.useState(null),[G,X]=l.useState(null),[J,m]=l.useState(null),$=l.useRef(null),q=l.useCallback(_=>{const Y=_.target.files?.[0];if(Y){b(Y);const Q=URL.createObjectURL(Y);R(Q),m(null),W(null),X(null);const u=document.createElement("video");u.onloadedmetadata=()=>{P({duration:u.duration.toFixed(1),width:u.videoWidth,height:u.videoHeight})},u.src=Q}},[]),le=l.useCallback(_=>{_.preventDefault();const Y=_.dataTransfer.files?.[0];if(Y&&Y.type.startsWith("video/")){b(Y);const Q=URL.createObjectURL(Y);R(Q),m(null),W(null),X(null);const u=document.createElement("video");u.onloadedmetadata=()=>{P({duration:u.duration.toFixed(1),width:u.videoWidth,height:u.videoHeight})},u.src=Q}},[]),F=async()=>{if(!d){N("Log in om te genereren");return}if(!p)return;const _=T!=="none"?gh[T]+(C?", "+C:""):C;if(!_.trim()){W("Please select a style or enter a prompt");return}fe(!0),W(null),X(null);try{const Y=new FormData;Y.append("file",p),Y.append("prompt",_),Y.append("negative_prompt",V),Y.append("denoise",String(D)),Y.append("fps",String(Z)),Y.append("max_frames",String(j)),Y.append("steps",String(v)),Y.append("cfg",String(re)),Y.append("seed",String(ge));const Q=await ht(`${ve}/generate-v2v`,Y);if(!Q.ok)throw new Error(Q.data?.detail||"V2V transform failed");const u=Q.data?.prompt_id;if(!u)throw new Error("No prompt_id returned");X({promptId:u,style:T!=="none"?T:"custom"}),x&&x({prompt_id:u})}catch(Y){console.error("V2V error:",Y),W(Y.message)}finally{fe(!1)}};return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(Xn,{size:18}),"Source Video"]}),t.jsxs("div",{className:`upload-dropzone ${S?"has-preview":""}`,onDrop:le,onDragOver:_=>_.preventDefault(),onClick:()=>document.getElementById("v2v-file-input").click(),children:[S?t.jsx("video",{ref:$,src:S,className:"upload-preview",controls:!0,muted:!0,loop:!0,style:{maxHeight:"250px"}}):t.jsxs("div",{className:"upload-placeholder",children:[t.jsx(pt,{size:32}),t.jsx("p",{children:"Drop video here or click to upload"}),t.jsx("span",{style:{fontSize:"12px",opacity:.6},children:"MP4, WebM, MOV"})]}),t.jsx("input",{id:"v2v-file-input",type:"file",accept:"video/*",onChange:q,style:{display:"none"}})]}),I&&t.jsxs("div",{className:"video-info",children:[t.jsxs("span",{children:["📐 ",I.width," × ",I.height,"px"]}),t.jsxs("span",{children:["⏱️ ",I.duration,"s"]})]})]}),t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(xr,{size:18}),"Style Transform"]}),t.jsx("div",{className:"style-grid",children:hh.map(_=>t.jsxs("button",{className:`style-btn ${T===_.value?"active":""}`,onClick:()=>A(_.value),children:[t.jsx("span",{className:"style-name",children:_.label}),t.jsx("span",{className:"style-desc",children:_.desc})]},_.value))})]}),t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:["Prompt ",T!=="none"&&t.jsx("span",{className:"hint",children:"(optional - adds to style)"})]}),t.jsx("textarea",{value:C,onChange:_=>L(_.target.value),placeholder:T!=="none"?"Add extra details to the style...":"Describe the desired look...",rows:3,className:"prompt-textarea"})]}),t.jsxs("div",{className:"tool-section",children:[t.jsx("h3",{children:"Transform Strength"}),t.jsxs("div",{className:"slider-row",children:[t.jsx("input",{type:"range",min:"0.1",max:"1",step:"0.05",value:D,onChange:_=>ee(parseFloat(_.target.value))}),t.jsxs("span",{className:"slider-value",children:[(D*100).toFixed(0),"%"]})]}),t.jsxs("div",{className:"slider-labels",children:[t.jsx("span",{children:"Subtle"}),t.jsx("span",{children:"Complete"})]})]}),t.jsxs("div",{className:"tool-section collapsible",children:[t.jsxs("h3",{onClick:()=>h(!B),style:{cursor:"pointer"},children:[t.jsx(Yn,{size:16}),"Advanced Settings",t.jsx(Qt,{size:16,style:{marginLeft:"auto",transform:B?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.2s"}})]}),B&&t.jsxs("div",{className:"advanced-content",children:[t.jsxs("div",{className:"form-row",children:[t.jsx("label",{children:"Output FPS"}),t.jsxs("select",{value:Z,onChange:_=>K(parseInt(_.target.value)),children:[t.jsx("option",{value:8,children:"8 fps"}),t.jsx("option",{value:12,children:"12 fps"}),t.jsx("option",{value:16,children:"16 fps"}),t.jsx("option",{value:24,children:"24 fps"})]})]}),t.jsxs("div",{className:"form-row",children:[t.jsx("label",{children:"Max Frames"}),t.jsxs("select",{value:j,onChange:_=>k(parseInt(_.target.value)),children:[t.jsx("option",{value:16,children:"16 frames (~2s @8fps)"}),t.jsx("option",{value:32,children:"32 frames (~4s @8fps)"}),t.jsx("option",{value:48,children:"48 frames (~6s @8fps)"}),t.jsx("option",{value:64,children:"64 frames (~8s @8fps)"})]})]}),t.jsxs("div",{className:"form-row",children:[t.jsx("label",{children:"Steps"}),t.jsx("input",{type:"number",min:10,max:50,value:v,onChange:_=>te(parseInt(_.target.value))})]}),t.jsxs("div",{className:"form-row",children:[t.jsx("label",{children:"CFG Scale"}),t.jsx("input",{type:"number",min:1,max:15,step:.5,value:re,onChange:_=>xe(parseFloat(_.target.value))})]}),t.jsxs("div",{className:"form-row",children:[t.jsx("label",{children:"Seed (-1 = random)"}),t.jsx("input",{type:"number",value:ge,onChange:_=>E(parseInt(_.target.value)||-1)})]}),t.jsxs("div",{className:"form-row",children:[t.jsx("label",{children:"Negative Prompt"}),t.jsx("textarea",{value:V,onChange:_=>U(_.target.value),rows:2,style:{fontSize:"12px"}})]})]})]}),G&&t.jsxs("div",{className:"queued-notice",children:["✅ Job queued! Check the Queue panel for progress.",t.jsx("span",{className:"queued-mode",children:G.style.toUpperCase()})]}),ie&&t.jsxs("div",{className:"error-message",children:["⚠️ ",ie]}),t.jsx("button",{className:"btn-primary btn-large",onClick:F,disabled:!p||ue,children:ue?t.jsxs(t.Fragment,{children:[t.jsx(at,{size:18,className:"spin"}),"Queueing..."]}):t.jsxs(t.Fragment,{children:[t.jsx(xr,{size:18}),"Transform Video"]})}),J&&t.jsxs("div",{className:"result-section",children:[t.jsx("h3",{children:"Result"}),t.jsx("video",{src:J,controls:!0,className:"result-video"}),t.jsx("a",{href:J,download:!0,className:"btn-secondary",style:{marginTop:12},children:"Download Video"})]}),t.jsx("style",{children:`
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
      `})]})}const yh=[{value:"brief",label:"Brief",desc:"Short 1-2 sentence description"},{value:"detailed",label:"Detailed",desc:"Comprehensive scene analysis"},{value:"prompt",label:"Prompt Style",desc:"Optimized for AI generation"},{value:"timeline",label:"Timeline",desc:"Frame-by-frame breakdown"}],bh=[{value:"smolvlm",label:"SmolVLM",desc:"Fast, lightweight vision model"},{value:"cogvlm",label:"CogVLM",desc:"High quality, slower"},{value:"llava",label:"LLaVA",desc:"Balanced quality/speed"}],jh=[{value:"upload",label:"Upload",icon:pt},{value:"youtube",label:"YouTube",icon:Sx}];function wh(){const[c,x]=l.useState("upload"),[d,N]=l.useState(null),[p,b]=l.useState(null),[S,R]=l.useState(null),[I,P]=l.useState(""),[T,A]=l.useState(null),[C,L]=l.useState(!1),[V,U]=l.useState(null),[D,ee]=l.useState("smolvlm"),[Z,K]=l.useState("detailed"),[j,k]=l.useState(1),[B,h]=l.useState(8),[v,te]=l.useState(!1),[re,xe]=l.useState(!1),[ge,E]=l.useState(null),[ue,fe]=l.useState(""),[ie,W]=l.useState(null),[G,X]=l.useState(!1),J=l.useRef(null),m=l.useCallback(u=>{const he=u.target.files?.[0];if(he){N(he);const ze=URL.createObjectURL(he);b(ze),W(null),E(null);const pe=document.createElement("video");pe.onloadedmetadata=()=>{R({duration:pe.duration.toFixed(1),width:pe.videoWidth,height:pe.videoHeight})},pe.src=ze}},[]),$=l.useCallback(u=>{u.preventDefault();const he=u.dataTransfer.files?.[0];if(he&&he.type.startsWith("video/")){N(he);const ze=URL.createObjectURL(he);b(ze),W(null),E(null),U(null);const pe=document.createElement("video");pe.onloadedmetadata=()=>{R({duration:pe.duration.toFixed(1),width:pe.videoWidth,height:pe.videoHeight})},pe.src=ze}},[]),q=u=>/^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+/.test(u),le=u=>{const he=u.target.value;P(he),A(null),E(null)},F=async()=>{if(!I||!q(I)){E("Please enter a valid YouTube URL");return}L(!0),E(null);try{const u=await Es(`${ve}/youtube/info`,{url:I});if(!u.ok)throw new Error(u.data?.detail||"Failed to fetch video info");A(u.data)}catch(u){E(u.message)}finally{L(!1)}},_=async()=>{if(I){L(!0),E(null),fe("Downloading video from YouTube...");try{const u=await Es(`${ve}/youtube/download`,{url:I,format:"video",quality:"720p"});if(!u.ok)throw new Error(u.data?.detail||"Failed to download video");U(u.data.path),b(`${ve}/file/${encodeURIComponent(u.data.path)}`),R({duration:u.data.duration?.toFixed(1)||T?.duration,width:u.data.width||T?.width||1280,height:u.data.height||T?.height||720,title:T?.title})}catch(u){E(u.message)}finally{L(!1),fe("")}}},Y=async()=>{if(!(!d&&!V)){xe(!0),E(null),fe("Uploading video...");try{const u=new FormData;V?u.append("video_path",V):u.append("file",d),u.append("model",D),u.append("mode",Z),u.append("frame_interval",String(j)),u.append("max_frames",String(B)),fe("Analyzing video...");const he=await ht(`${ve}/caption-video`,u);if(!he.ok)throw new Error(he.data?.detail||"Video analysis failed");W(he.data)}catch(u){console.error("V2T error:",u),E(u.message)}finally{xe(!1),fe("")}}},Q=async u=>{await navigator.clipboard.writeText(u),X(!0),setTimeout(()=>X(!1),2e3)};return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(Xn,{size:18}),"Source Video"]}),t.jsx("div",{className:"source-tabs",children:jh.map(u=>t.jsxs("button",{className:`source-tab ${c===u.value?"active":""}`,onClick:()=>{x(u.value),E(null)},children:[t.jsx(u.icon,{size:16}),u.label]},u.value))}),c==="upload"&&t.jsxs("div",{className:`upload-dropzone ${p?"has-preview":""}`,onDrop:$,onDragOver:u=>u.preventDefault(),onClick:()=>document.getElementById("v2t-file-input").click(),children:[p&&!V?t.jsx("video",{ref:J,src:p,className:"upload-preview",controls:!0,muted:!0,style:{maxHeight:"200px"}}):t.jsxs("div",{className:"upload-placeholder",children:[t.jsx(pt,{size:32}),t.jsx("p",{children:"Drop video here or click to upload"}),t.jsx("span",{style:{fontSize:"12px",opacity:.6},children:"MP4, WebM, MOV"})]}),t.jsx("input",{id:"v2t-file-input",type:"file",accept:"video/*",onChange:m,style:{display:"none"}})]}),c==="youtube"&&t.jsxs("div",{className:"youtube-section",children:[t.jsxs("div",{className:"youtube-input-row",children:[t.jsxs("div",{className:"youtube-input-wrapper",children:[t.jsx(mu,{size:16,className:"youtube-input-icon"}),t.jsx("input",{type:"text",className:"youtube-input",placeholder:"Paste YouTube URL here...",value:I,onChange:le,onKeyDown:u=>u.key==="Enter"&&F()})]}),t.jsx("button",{className:"btn-secondary",onClick:F,disabled:C||!I,children:C?t.jsx(at,{size:16,className:"spin"}):"Fetch"})]}),T&&t.jsxs("div",{className:"youtube-preview",children:[T.thumbnail&&t.jsx("img",{src:T.thumbnail,alt:"thumbnail",className:"youtube-thumbnail"}),t.jsxs("div",{className:"youtube-info",children:[t.jsx("span",{className:"youtube-title",children:T.title}),t.jsxs("span",{className:"youtube-meta",children:[T.channel," • ",T.duration,"s • ",T.view_count?.toLocaleString()," views"]})]}),t.jsx("button",{className:"btn-primary",onClick:_,disabled:C,children:C?t.jsx(at,{size:16,className:"spin"}):t.jsxs(t.Fragment,{children:[t.jsx(qt,{size:16}),"Download"]})})]}),V&&t.jsxs("div",{className:"youtube-downloaded",children:[t.jsx(mr,{size:16,style:{color:"#22c55e"}}),t.jsx("span",{children:"Video ready for analysis"}),p&&t.jsx("video",{src:p,className:"upload-preview",controls:!0,muted:!0,style:{maxHeight:"200px",marginTop:"12px",width:"100%"}})]})]}),S&&t.jsxs("div",{className:"video-info",children:[t.jsxs("span",{children:["📐 ",S.width," × ",S.height]}),t.jsxs("span",{children:["⏱️ ",S.duration,"s"]})]})]}),t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(Bl,{size:18}),"Analysis Model"]}),t.jsx("div",{className:"model-grid",children:bh.map(u=>t.jsxs("button",{className:`model-btn ${D===u.value?"active":""}`,onClick:()=>ee(u.value),children:[t.jsx("span",{className:"model-name",children:u.label}),t.jsx("span",{className:"model-desc",children:u.desc})]},u.value))})]}),t.jsxs("div",{className:"tool-section",children:[t.jsx("h3",{children:"Output Style"}),t.jsx("div",{className:"mode-grid",children:yh.map(u=>t.jsxs("button",{className:`mode-btn ${Z===u.value?"active":""}`,onClick:()=>K(u.value),children:[t.jsx("span",{className:"mode-name",children:u.label}),t.jsx("span",{className:"mode-desc",children:u.desc})]},u.value))})]}),t.jsxs("div",{className:"tool-section collapsible",children:[t.jsxs("h3",{onClick:()=>te(!v),style:{cursor:"pointer"},children:[t.jsx(Yn,{size:16}),"Advanced",t.jsx(Qt,{size:16,style:{marginLeft:"auto",transform:v?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.2s"}})]}),v&&t.jsxs("div",{className:"advanced-content",children:[t.jsxs("div",{className:"form-row",children:[t.jsx("label",{children:"Frame Interval"}),t.jsxs("select",{value:j,onChange:u=>k(parseFloat(u.target.value)),children:[t.jsx("option",{value:.5,children:"Every 0.5s"}),t.jsx("option",{value:1,children:"Every 1s"}),t.jsx("option",{value:2,children:"Every 2s"}),t.jsx("option",{value:5,children:"Every 5s"})]})]}),t.jsxs("div",{className:"form-row",children:[t.jsx("label",{children:"Max Frames"}),t.jsxs("select",{value:B,onChange:u=>h(parseInt(u.target.value)),children:[t.jsx("option",{value:4,children:"4 frames"}),t.jsx("option",{value:8,children:"8 frames"}),t.jsx("option",{value:16,children:"16 frames"}),t.jsx("option",{value:32,children:"32 frames"})]})]})]})]}),ge&&t.jsxs("div",{className:"error-message",children:["⚠️ ",ge]}),t.jsx("button",{className:"btn-primary btn-large",onClick:Y,disabled:!d&&!V||re,children:re?t.jsxs(t.Fragment,{children:[t.jsx(at,{size:18,className:"spin"}),ue]}):t.jsxs(t.Fragment,{children:[t.jsx(Bl,{size:18}),"Analyze Video"]})}),ie&&t.jsxs("div",{className:"result-section",children:[t.jsxs("div",{className:"result-header",children:[t.jsx("h3",{children:"Description"}),t.jsxs("button",{className:"copy-btn",onClick:()=>Q(ie.caption||ie.description),children:[G?t.jsx(mr,{size:16}):t.jsx(un,{size:16}),G?"Copied!":"Copy"]})]}),t.jsx("div",{className:"result-text",children:ie.caption||ie.description}),ie.timeline&&ie.timeline.length>0&&t.jsxs("div",{className:"timeline-section",children:[t.jsx("h4",{children:"Timeline"}),ie.timeline.map((u,he)=>t.jsxs("div",{className:"timeline-item",children:[t.jsxs("span",{className:"timeline-time",children:[u.time,"s"]}),t.jsx("span",{className:"timeline-desc",children:u.description})]},he))]}),ie.prompt&&t.jsxs("div",{className:"prompt-section",children:[t.jsxs("div",{className:"prompt-header",children:[t.jsx("h4",{children:"AI Generation Prompt"}),t.jsx("button",{className:"copy-btn small",onClick:()=>Q(ie.prompt),children:t.jsx(un,{size:14})})]}),t.jsx("div",{className:"prompt-text",children:ie.prompt})]})]}),t.jsx("style",{children:`
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
      `})]})}const kh=["video/mp4","video/webm","video/quicktime"],Sh=[{id:"f5v1",label:"F5-TTS v1",description:"Fast, high quality"},{id:"e2",label:"E2-TTS",description:"More expressive"}],Nh=[{id:"custom",label:"Upload Voice Sample",isCustom:!0},{id:"alloy",label:"Alloy (Neutral)"},{id:"echo",label:"Echo (Male)"},{id:"fable",label:"Fable (British)"},{id:"onyx",label:"Onyx (Deep Male)"},{id:"nova",label:"Nova (Female)"},{id:"shimmer",label:"Shimmer (Soft Female)"}];function Ch({onOutput:c,onJobSubmitted:x}){const{user:d,requestLogin:N}=Ke(),[p,b]=l.useState(null),[S,R]=l.useState(null),[I,P]=l.useState(null),[T,A]=l.useState(""),[C,L]=l.useState("f5v1"),[V,U]=l.useState("nova"),[D,ee]=l.useState(null),[Z,K]=l.useState(null),[j,k]=l.useState(1.5),[B,h]=l.useState(20),[v,te]=l.useState(!1),[re,xe]=l.useState(!1),[ge,E]=l.useState(!1),[ue,fe]=l.useState(null),[ie,W]=l.useState(null),[G,X]=l.useState(null),J=l.useRef(null),m=l.useRef(null),$=l.useRef(null),q=l.useCallback(u=>{u.preventDefault();const he=u.dataTransfer?.files?.[0]||u.target?.files?.[0];he&&kh.some(ze=>he.type.includes(ze.split("/")[1]))?(b(he),R(URL.createObjectURL(he)),P(null),W(null),X(null)):he&&W("Please upload a valid video file (MP4, WebM)")},[]),le=l.useCallback(u=>{u.preventDefault();const he=u.dataTransfer?.files?.[0]||u.target?.files?.[0];he&&he.type.startsWith("audio/")?(ee(he),K(URL.createObjectURL(he)),W(null)):he&&W("Please upload a valid audio file for voice sample")},[]),F=async u=>{const he=new FormData;he.append("file",u);try{const ze=await ht(`${ve}/upload`,he);if(ze.ok&&ze.data?.path)return ze.data.path;throw new Error(ze.data?.detail||"Upload failed")}catch(ze){throw new Error(`Upload failed: ${ze.message}`)}},_=async()=>{if(!d){N("Log in om te genereren");return}if(!p||!T.trim()){W("Please upload a video and enter text");return}xe(!0),W(null),X(null);try{fe("Uploading video..."),E(!0);let u=I;u||(u=await F(p),P(u));let he=null;V==="custom"&&D&&(fe("Uploading voice sample..."),he=await F(D)),E(!1),fe("Generating speech...");const ze=new FormData;ze.append("text",T),ze.append("model",C),V==="custom"&&he?ze.append("voice_sample",he):V!=="custom"&&ze.append("voice_preset",V);const pe=await ht(`${ve}/voice-clone`,ze);if(!pe.ok)throw new Error(pe.data?.detail||"TTS generation failed");const Ne=pe.data?.path||pe.data?.audio_path;if(!Ne)throw new Error("TTS did not return audio path");fe("Applying lip sync...");const Re={video_path:u,audio_path:Ne,lips_expression:j,inference_steps:B,seed:-1},Ve=await Es(`${ve}/lip-sync`,Re);if(!Ve.ok)throw new Error(Ve.data?.detail||"Lip sync failed");X({promptId:Ve.data?.prompt_id,text:T.slice(0,30)+(T.length>30?"...":"")}),x&&x({prompt_id:Ve.data?.prompt_id})}catch(u){console.error("❌ Speech-to-Video error:",u),W(u.message)}finally{xe(!1),E(!1),fe(null)}},Y=()=>{b(null),R(null),P(null),X(null)},Q=()=>{ee(null),K(null)};return t.jsxs("div",{className:"tool-container space-y-4 p-4",children:[t.jsxs("div",{className:"text-center mb-4",children:[t.jsxs("h2",{className:"text-xl font-bold text-white flex items-center justify-center gap-2",children:[t.jsx(Rl,{className:"w-6 h-6 text-purple-400"}),"Speech to Video"]}),t.jsx("p",{className:"text-gray-400 text-sm mt-1",children:"Generate speech from text and sync it to a video"})]}),t.jsxs("div",{onClick:()=>m.current?.click(),onDrop:q,onDragOver:u=>u.preventDefault(),className:"border-2 border-dashed border-gray-600 rounded-lg p-6 text-center cursor-pointer hover:border-purple-500 transition-colors",children:[t.jsx("input",{ref:m,type:"file",accept:"video/*",onChange:q,className:"hidden"}),S?t.jsxs("div",{className:"space-y-2",children:[t.jsx("video",{ref:J,src:S,className:"max-h-40 mx-auto rounded",controls:!0,muted:!0}),t.jsxs("div",{className:"flex items-center justify-center gap-2",children:[t.jsx("span",{className:"text-sm text-gray-400",children:p?.name}),t.jsx("button",{onClick:u=>{u.stopPropagation(),Y()},className:"p-1 text-red-400 hover:text-red-300",children:t.jsx(lt,{className:"w-4 h-4"})})]})]}):t.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[t.jsx(Xn,{className:"w-10 h-10"}),t.jsx("span",{children:"Drop video here or click to upload"}),t.jsx("span",{className:"text-xs",children:"MP4, WebM supported"})]})]}),t.jsxs("div",{children:[t.jsxs("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:[t.jsx(Rl,{className:"w-4 h-4 inline mr-1"}),"Text to Speak"]}),t.jsx("textarea",{value:T,onChange:u=>A(u.target.value),placeholder:"Enter the text you want the character to say...",className:"w-full px-3 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 resize-none",rows:4}),t.jsxs("div",{className:"text-xs text-gray-500 mt-1 text-right",children:[T.length," characters"]})]}),t.jsxs("div",{children:[t.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"TTS Model"}),t.jsx("div",{className:"grid grid-cols-2 gap-2",children:Sh.map(u=>t.jsxs("button",{onClick:()=>L(u.id),className:`px-3 py-2 text-sm rounded transition-colors text-left ${C===u.id?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:[t.jsx("div",{className:"font-medium",children:u.label}),t.jsx("div",{className:"text-xs opacity-70",children:u.description})]},u.id))})]}),t.jsxs("div",{children:[t.jsxs("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:[t.jsx(Xl,{className:"w-4 h-4 inline mr-1"}),"Voice"]}),t.jsx("select",{value:V,onChange:u=>U(u.target.value),className:"w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white",children:Nh.map(u=>t.jsx("option",{value:u.id,children:u.label},u.id))})]}),V==="custom"&&t.jsxs("div",{onClick:()=>$.current?.click(),onDrop:le,onDragOver:u=>u.preventDefault(),className:"border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-purple-500 transition-colors",children:[t.jsx("input",{ref:$,type:"file",accept:"audio/*",onChange:le,className:"hidden"}),Z?t.jsxs("div",{className:"space-y-2",children:[t.jsx("audio",{src:Z,controls:!0,className:"mx-auto"}),t.jsxs("div",{className:"flex items-center justify-center gap-2",children:[t.jsx("span",{className:"text-sm text-gray-400",children:D?.name}),t.jsx("button",{onClick:u=>{u.stopPropagation(),Q()},className:"p-1 text-red-400 hover:text-red-300",children:t.jsx(lt,{className:"w-4 h-4"})})]})]}):t.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[t.jsx(Br,{className:"w-6 h-6"}),t.jsx("span",{className:"text-sm",children:"Upload voice sample (5-15 sec recommended)"})]})]}),t.jsxs("div",{className:"border border-gray-700 rounded-lg overflow-hidden",children:[t.jsxs("button",{onClick:()=>te(!v),className:"w-full px-4 py-2 bg-gray-800 flex items-center justify-between text-gray-300 hover:bg-gray-750",children:[t.jsxs("span",{className:"text-sm font-medium flex items-center gap-2",children:[t.jsx(hu,{className:"w-4 h-4"}),"Lip Sync Settings"]}),t.jsx(Qt,{className:`w-4 h-4 transition-transform ${v?"rotate-180":""}`})]}),v&&t.jsxs("div",{className:"p-4 space-y-4 bg-gray-850",children:[t.jsxs("div",{children:[t.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Lips Expression: ",j.toFixed(1)]}),t.jsx("input",{type:"range",min:.5,max:3,step:.1,value:j,onChange:u=>k(parseFloat(u.target.value)),className:"w-full accent-purple-500"}),t.jsx("span",{className:"text-xs text-gray-500",children:"Higher = more pronounced lip movement"})]}),t.jsxs("div",{children:[t.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Inference Steps: ",B]}),t.jsx("input",{type:"range",min:10,max:50,step:5,value:B,onChange:u=>h(parseInt(u.target.value)),className:"w-full accent-purple-500"}),t.jsx("span",{className:"text-xs text-gray-500",children:"Higher = better quality, slower"})]})]})]}),t.jsx("button",{onClick:_,disabled:re||!p||!T.trim(),className:"w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors",children:re?t.jsxs(t.Fragment,{children:[t.jsx(at,{className:"w-5 h-5 animate-spin"}),ue||"Processing..."]}):t.jsxs(t.Fragment,{children:[t.jsx(Rl,{className:"w-5 h-5"}),"Generate Speech Video"]})}),G&&t.jsxs("div",{className:"p-3 bg-green-900/50 border border-green-700 rounded-lg text-green-200 text-sm",children:['✅ Speech-to-Video queued! "',G.text,'" - Check queue panel for progress']}),ie&&t.jsx("div",{className:"p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm",children:ie}),t.jsx("div",{className:"text-xs text-gray-500 text-center",children:"This tool generates speech from your text using TTS, then applies lip sync to match the video."})]})}const Kd=[{value:"realesrgan-video",label:"Real-ESRGAN Video",desc:"AI-enhanced video upscaling",scale:[2,4]},{value:"basic-lanczos",label:"Basic Lanczos",desc:"Fast traditional upscaling",scale:[2,4]}];function _h({onOutput:c,onJobSubmitted:x}){const{user:d,requestLogin:N}=Ke(),[p,b]=l.useState(null),[S,R]=l.useState(null),[I,P]=l.useState(null),[T,A]=l.useState("realesrgan-video"),[C,L]=l.useState(!1),[V,U]=l.useState(null),[D,ee]=l.useState(null),Z=l.useCallback(k=>{const B=k.target.files?.[0];if(B){b(B);const h=URL.createObjectURL(B);R(h),setResult(null),U(null),ee(null);const v=document.createElement("video");v.onloadedmetadata=()=>{P({duration:v.duration.toFixed(1),width:v.videoWidth,height:v.videoHeight})},v.src=h}},[]),K=l.useCallback(k=>{k.preventDefault();const B=k.dataTransfer.files?.[0];if(B&&B.type.startsWith("video/")){b(B);const h=URL.createObjectURL(B);R(h),U(null),ee(null);const v=document.createElement("video");v.onloadedmetadata=()=>{P({duration:v.duration.toFixed(1),width:v.videoWidth,height:v.videoHeight})},v.src=h}},[]),j=async()=>{if(!d){N("Log in om te genereren");return}if(p){L(!0),U(null),ee(null);try{const k=new FormData;k.append("file",p),k.append("model",T);const B=await ht(`${ve}/upscale-video`,k);if(!B.ok)throw new Error(B.data?.detail||"Video upscaling failed");const h=B.data?.prompt_id;if(!h)throw new Error("No prompt_id returned");ee({promptId:h,model:Kd.find(v=>v.value===T)?.label||T}),x&&x(h)}catch(k){console.error("Video upscale error:",k),U(k.message||"Failed to upscale video")}finally{L(!1)}}};return t.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"16px",padding:"20px"},children:[t.jsxs("div",{style:{marginBottom:"8px"},children:[t.jsx("h2",{style:{fontSize:"1.3rem",fontWeight:600,marginBottom:"4px"},children:"Video Upscaler"}),t.jsx("p",{style:{fontSize:"0.85rem",color:"var(--text-muted)"},children:"AI-enhanced video upscaling • 480p → 720p → 1080p → 4K"})]}),t.jsxs("div",{onDrop:K,onDragOver:k=>k.preventDefault(),style:{border:"2px dashed var(--border-color)",borderRadius:"8px",padding:"24px",textAlign:"center",cursor:"pointer",transition:"all 0.2s"},onClick:()=>document.getElementById("video-upscale-file")?.click(),children:[t.jsx(pt,{size:32,style:{margin:"0 auto 12px",color:"var(--text-muted)"}}),t.jsx("p",{style:{fontSize:"0.9rem",color:"var(--text-secondary)",marginBottom:"4px"},children:p?p.name:"Drop video or click to upload"}),I&&t.jsxs("p",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:[I.width,"×",I.height," • ",I.duration,"s"]}),t.jsx("input",{id:"video-upscale-file",type:"file",accept:"video/*",onChange:Z,style:{display:"none"}})]}),S&&t.jsx("div",{style:{borderRadius:"8px",overflow:"hidden",maxWidth:"100%"},children:t.jsx("video",{src:S,controls:!0,style:{width:"100%",maxHeight:"400px",display:"block"}})}),t.jsx("div",{style:{display:"flex",flexDirection:"column",gap:"12px"},children:t.jsxs("div",{children:[t.jsx("label",{style:{display:"block",fontSize:"0.85rem",marginBottom:"6px",color:"var(--text-secondary)"},children:"Upscale Model"}),t.jsx("select",{value:T,onChange:k=>A(k.target.value),style:{width:"100%",padding:"8px 12px",borderRadius:"6px",border:"1px solid var(--border-color)",background:"var(--bg-secondary)",color:"var(--text-primary)",fontSize:"0.9rem"},children:Kd.map(k=>t.jsxs("option",{value:k.value,children:[k.label," - ",k.desc]},k.value))}),t.jsx("p",{style:{fontSize:"0.7rem",color:"var(--text-muted)",marginTop:"4px"},children:"Note: Currently uses fixed 4x upscaling with RealESRGAN. Custom resolution and quality settings coming soon."})]})}),V&&t.jsx("div",{style:{padding:"12px",background:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.3)",borderRadius:"6px"},children:t.jsx("p",{style:{fontSize:"0.85rem",color:"#ef4444"},children:V})}),D&&t.jsxs("div",{style:{padding:"12px",background:"rgba(34, 197, 94, 0.1)",border:"1px solid rgba(34, 197, 94, 0.3)",borderRadius:"6px"},children:[t.jsxs("p",{style:{fontSize:"0.85rem",color:"#22c55e"},children:["✓ Video upscale queued! (",D.model,")"]}),t.jsxs("p",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"4px"},children:["Job ID: ",D.promptId]})]}),t.jsx("button",{onClick:j,disabled:!p||C,style:{padding:"14px",borderRadius:"8px",border:"none",background:!p||C?"var(--bg-tertiary)":"var(--accent-color)",color:!p||C?"var(--text-muted)":"white",fontSize:"1rem",fontWeight:600,cursor:!p||C?"not-allowed":"pointer",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px",transition:"all 0.2s"},children:C?t.jsxs(t.Fragment,{children:[t.jsx(at,{size:20,style:{animation:"spin 1s linear infinite"}}),"Upscaling..."]}):t.jsxs(t.Fragment,{children:[t.jsx(Hl,{size:20}),"Upscale Video"]})})]})}const Jd=[{value:"rife",label:"RIFE",desc:"Fast & high quality",recommended:!0},{value:"film",label:"FILM",desc:"Google Research model",recommended:!1}],Fl=[{from:15,to:30,label:"15fps → 30fps (2x)",multiplier:2},{from:15,to:60,label:"15fps → 60fps (4x)",multiplier:4},{from:24,to:30,label:"24fps → 30fps (1.25x)",multiplier:1.25},{from:24,to:60,label:"24fps → 60fps (2.5x)",multiplier:2.5},{from:30,to:60,label:"30fps → 60fps (2x)",multiplier:2}],Dl=[{value:"2x",label:"2x Slower",multiplier:2,desc:"Double frame count"},{value:"4x",label:"4x Slower",multiplier:4,desc:"Quadruple frame count"},{value:"8x",label:"8x Slower",multiplier:8,desc:"Epic slow motion"}];function zh({onOutput:c,onJobSubmitted:x}){const{user:d,requestLogin:N}=Ke(),[p,b]=l.useState(null),[S,R]=l.useState(null),[I,P]=l.useState(null),[T,A]=l.useState("rife"),[C,L]=l.useState("fps"),[V,U]=l.useState("30fps → 60fps (2x)"),[D,ee]=l.useState("2x"),[Z,K]=l.useState(!1),[j,k]=l.useState(null),[B,h]=l.useState(null),[v,te]=l.useState(null),re=l.useCallback(E=>{const ue=E.target.files?.[0];if(ue){b(ue);const fe=URL.createObjectURL(ue);R(fe),te(null),k(null),h(null);const ie=document.createElement("video");ie.onloadedmetadata=()=>{P({duration:ie.duration.toFixed(1),width:ie.videoWidth,height:ie.videoHeight,fps:30})},ie.src=fe}},[]),xe=l.useCallback(E=>{E.preventDefault();const ue=E.dataTransfer.files?.[0];if(ue&&ue.type.startsWith("video/")){b(ue);const fe=URL.createObjectURL(ue);R(fe),te(null),k(null),h(null);const ie=document.createElement("video");ie.onloadedmetadata=()=>{P({duration:ie.duration.toFixed(1),width:ie.videoWidth,height:ie.videoHeight,fps:30})},ie.src=fe}},[]),ge=async()=>{if(!d){N("Log in om te genereren");return}if(p){K(!0),k(null),h(null);try{const E=new FormData;if(E.append("file",p),E.append("model",T),E.append("mode",C),C==="fps"){const ie=Fl.find(W=>W.label===V);E.append("target_fps",String(ie?.to||60)),E.append("multiplier",String(ie?.multiplier||2))}else{const ie=Dl.find(W=>W.value===D);E.append("multiplier",String(ie?.multiplier||2))}const ue=await ht(`${ve}/interpolate-video`,E);if(!ue.ok)throw new Error(ue.data?.detail||"Frame interpolation failed");const fe=ue.data?.prompt_id;if(!fe)throw new Error("No prompt_id returned");h({promptId:fe,model:Jd.find(ie=>ie.value===T)?.label||T,preset:C==="fps"?V:`${D} Slow Motion`}),x&&x(fe)}catch(E){console.error("Interpolation error:",E),k(E.message||"Failed to interpolate video")}finally{K(!1)}}};return t.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"16px",padding:"20px"},children:[t.jsxs("div",{style:{marginBottom:"8px"},children:[t.jsx("h2",{style:{fontSize:"1.3rem",fontWeight:600,marginBottom:"4px"},children:"Frame Interpolation"}),t.jsx("p",{style:{fontSize:"0.85rem",color:"var(--text-muted)"},children:"Increase FPS & create smooth slow motion • RIFE/FILM integration"})]}),t.jsxs("div",{onDrop:xe,onDragOver:E=>E.preventDefault(),style:{border:"2px dashed var(--border-color)",borderRadius:"8px",padding:"24px",textAlign:"center",cursor:"pointer",transition:"all 0.2s"},onClick:()=>document.getElementById("interpolate-file")?.click(),children:[t.jsx(pt,{size:32,style:{margin:"0 auto 12px",color:"var(--text-muted)"}}),t.jsx("p",{style:{fontSize:"0.9rem",color:"var(--text-secondary)",marginBottom:"4px"},children:p?p.name:"Drop video or click to upload"}),I&&t.jsxs("p",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:[I.width,"×",I.height," • ",I.duration,"s • ~",I.fps,"fps"]}),t.jsx("input",{id:"interpolate-file",type:"file",accept:"video/*",onChange:re,style:{display:"none"}})]}),S&&t.jsx("div",{style:{borderRadius:"8px",overflow:"hidden",maxWidth:"100%"},children:t.jsx("video",{src:S,controls:!0,style:{width:"100%",maxHeight:"400px",display:"block"}})}),t.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"12px"},children:[t.jsxs("div",{children:[t.jsx("label",{style:{display:"block",fontSize:"0.85rem",marginBottom:"6px",color:"var(--text-secondary)"},children:"Interpolation Model"}),t.jsx("div",{style:{display:"flex",gap:"8px"},children:Jd.map(E=>t.jsxs("button",{onClick:()=>A(E.value),type:"button",style:{flex:1,padding:"10px",borderRadius:"6px",border:T===E.value?"1px solid var(--accent-color)":"1px solid var(--border-color)",background:T===E.value?"rgba(59, 130, 246, 0.2)":"var(--bg-secondary)",color:T===E.value?"var(--accent-color)":"var(--text-secondary)",fontSize:"0.85rem",cursor:"pointer",transition:"all 0.15s"},children:[t.jsxs("div",{style:{fontWeight:600},children:[E.label," ",E.recommended&&"⭐"]}),t.jsx("div",{style:{fontSize:"0.7rem",marginTop:"2px",opacity:.8},children:E.desc})]},E.value))})]}),t.jsxs("div",{children:[t.jsx("label",{style:{display:"block",fontSize:"0.85rem",marginBottom:"6px",color:"var(--text-secondary)"},children:"Mode"}),t.jsxs("div",{style:{display:"flex",gap:"8px"},children:[t.jsxs("button",{onClick:()=>L("fps"),type:"button",style:{flex:1,padding:"10px",borderRadius:"6px",border:C==="fps"?"1px solid var(--accent-color)":"1px solid var(--border-color)",background:C==="fps"?"rgba(59, 130, 246, 0.2)":"var(--bg-secondary)",color:C==="fps"?"var(--accent-color)":"var(--text-secondary)",fontSize:"0.85rem",cursor:"pointer",transition:"all 0.15s"},children:[t.jsx("div",{style:{fontWeight:600},children:"FPS Conversion"}),t.jsx("div",{style:{fontSize:"0.7rem",marginTop:"2px",opacity:.8},children:"Increase frame rate"})]}),t.jsxs("button",{onClick:()=>L("slowmo"),type:"button",style:{flex:1,padding:"10px",borderRadius:"6px",border:C==="slowmo"?"1px solid var(--accent-color)":"1px solid var(--border-color)",background:C==="slowmo"?"rgba(59, 130, 246, 0.2)":"var(--bg-secondary)",color:C==="slowmo"?"var(--accent-color)":"var(--text-secondary)",fontSize:"0.85rem",cursor:"pointer",transition:"all 0.15s"},children:[t.jsx("div",{style:{fontWeight:600},children:"Slow Motion"}),t.jsx("div",{style:{fontSize:"0.7rem",marginTop:"2px",opacity:.8},children:"Smooth slow-mo"})]})]})]}),C==="fps"&&t.jsxs("div",{children:[t.jsx("label",{style:{display:"block",fontSize:"0.85rem",marginBottom:"6px",color:"var(--text-secondary)"},children:"Target FPS"}),t.jsx("div",{style:{display:"flex",flexWrap:"wrap",gap:"8px"},children:Fl.map(E=>t.jsx("button",{onClick:()=>U(E.label),type:"button",style:{padding:"8px 14px",borderRadius:"6px",border:V===E.label?"1px solid var(--accent-color)":"1px solid var(--border-color)",background:V===E.label?"rgba(59, 130, 246, 0.2)":"var(--bg-secondary)",color:V===E.label?"var(--accent-color)":"var(--text-secondary)",fontSize:"0.85rem",cursor:"pointer",transition:"all 0.15s"},children:E.label},E.label))})]}),C==="slowmo"&&t.jsxs("div",{children:[t.jsx("label",{style:{display:"block",fontSize:"0.85rem",marginBottom:"6px",color:"var(--text-secondary)"},children:"Slow Motion Speed"}),t.jsx("div",{style:{display:"flex",gap:"8px"},children:Dl.map(E=>t.jsxs("button",{onClick:()=>ee(E.value),type:"button",style:{flex:1,padding:"10px",borderRadius:"6px",border:D===E.value?"1px solid var(--accent-color)":"1px solid var(--border-color)",background:D===E.value?"rgba(59, 130, 246, 0.2)":"var(--bg-secondary)",color:D===E.value?"var(--accent-color)":"var(--text-secondary)",fontSize:"0.85rem",cursor:"pointer",transition:"all 0.15s"},title:E.desc,children:[t.jsx("div",{style:{fontWeight:600},children:E.label}),t.jsx("div",{style:{fontSize:"0.7rem",marginTop:"2px",opacity:.8},children:E.desc})]},E.value))})]})]}),j&&t.jsx("div",{style:{padding:"12px",background:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.3)",borderRadius:"6px"},children:t.jsx("p",{style:{fontSize:"0.85rem",color:"#ef4444"},children:j})}),B&&t.jsxs("div",{style:{padding:"12px",background:"rgba(34, 197, 94, 0.1)",border:"1px solid rgba(34, 197, 94, 0.3)",borderRadius:"6px"},children:[t.jsxs("p",{style:{fontSize:"0.85rem",color:"#22c55e"},children:["✓ Interpolation queued! (",B.model,", ",B.preset,")"]}),t.jsxs("p",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"4px"},children:["Job ID: ",B.promptId]})]}),t.jsx("button",{onClick:ge,disabled:!p||Z,style:{padding:"14px",borderRadius:"8px",border:"none",background:!p||Z?"var(--bg-tertiary)":"var(--accent-color)",color:!p||Z?"var(--text-muted)":"white",fontSize:"1rem",fontWeight:600,cursor:!p||Z?"not-allowed":"pointer",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px",transition:"all 0.2s"},children:Z?t.jsxs(t.Fragment,{children:[t.jsx(at,{size:20,style:{animation:"spin 1s linear infinite"}}),"Interpolating..."]}):t.jsxs(t.Fragment,{children:[t.jsx(Kl,{size:20}),"Interpolate Frames"]})}),v&&t.jsxs("div",{children:[t.jsxs("h3",{style:{fontSize:"1rem",marginBottom:"8px"},children:["Result (",C==="fps"?Fl.find(E=>E.label===V)?.label:Dl.find(E=>E.value===D)?.label,")"]}),t.jsx("div",{style:{borderRadius:"8px",overflow:"hidden"},children:t.jsx("video",{src:v,controls:!0,style:{width:"100%",display:"block"}})})]})]})}function Eh(){const[c,x]=l.useState([{id:1,name:"Text Generation",status:"completed",description:"Generate prompt from keywords"},{id:2,name:"Text to Image",status:"ready",description:"Create base image"},{id:3,name:"Image to Video",status:"pending",description:"Animate the image"},{id:4,name:"Upscale",status:"pending",description:"Enhance resolution"}]),[d,N]=l.useState(2);return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"grok-card",children:[t.jsxs("div",{className:"grok-card-header",children:[t.jsx("div",{className:"grok-card-title",children:"Production Pipeline"}),t.jsx(jx,{size:16,className:"text-muted"})]}),t.jsx("div",{style:{display:"flex",flexDirection:"column",gap:"16px"},children:c.map((p,b)=>t.jsxs("div",{className:`pipeline-step ${d===p.id?"active":""}`,style:{display:"flex",alignItems:"center",gap:"16px",padding:"16px",backgroundColor:d===p.id?"#1a1a1a":"transparent",borderRadius:"8px",border:d===p.id?"1px solid var(--border-color)":"1px solid transparent",opacity:p.status==="pending"?.5:1},children:[t.jsx("div",{style:{width:"32px",height:"32px",borderRadius:"50%",backgroundColor:p.status==="completed"?"#22c55e":d===p.id?"var(--text-primary)":"#333",color:p.status==="completed"||d===p.id?"var(--bg-root)":"var(--text-secondary)",display:"flex",alignItems:"center",justifyContent:"center",fontWeight:"bold",fontSize:"0.9rem"},children:p.status==="completed"?t.jsx(Yf,{size:18}):p.id}),t.jsxs("div",{style:{flex:1},children:[t.jsx("div",{style:{fontWeight:600,color:"var(--text-primary)"},children:p.name}),t.jsx("div",{style:{fontSize:"0.85rem",color:"var(--text-secondary)"},children:p.description})]}),b<c.length-1&&t.jsx(Lf,{size:16,className:"text-muted",style:{opacity:.3}})]},p.id))})]}),t.jsxs("div",{className:"grok-card",children:[t.jsx("div",{className:"grok-card-header",children:t.jsxs("div",{className:"grok-card-title",children:["Step Configuration: ",c.find(p=>p.id===d)?.name]})}),t.jsx("div",{className:"placeholder-state",style:{padding:"20px 0"},children:t.jsx("div",{className:"text-muted",children:"Configuration options for this step would appear here."})})]}),t.jsxs("button",{className:"primary-btn",style:{display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:[t.jsx(Va,{size:18}),"Run Pipeline"]})]})}function Ih({onOutput:c}){const{user:x,requestLogin:d}=Ke(),N=l.useRef(null),[p,b]=l.useState([]),[S,R]=l.useState(""),[I,P]=l.useState(10),[T,A]=l.useState(1e-4),[C,L]=l.useState(!1),[V,U]=l.useState(""),D=l.useMemo(()=>p.length>0&&S.trim().length>0&&!C,[p,S,C]),ee=j=>{const k=Array.from(j||[]);b(k),U("")},Z=()=>{b([]),N.current&&(N.current.value="")},K=async()=>{if(!x){d("Log in om te genereren");return}if(p.length===0){U("At least one image is required");return}if(!S.trim()){U("Model name is required");return}L(!0),U("");const j=new FormData;p.forEach(k=>j.append("files",k)),j.append("model_name",S.trim()),j.append("num_epochs",String(I)),j.append("learning_rate",String(T));try{const k=await ht(`${ve}/train-lora`,j);if(!k.ok){U(k.data?.detail||`Training failed (status ${k.status})`);return}c({kind:"lora",...k.data})}catch(k){const B=k?.message||"Failed to start LoRA training";U(B),await Wa({level:"error",message:"LoRA training failed",timestamp:new Date().toISOString(),meta:{message:B}})}finally{L(!1)}};return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"grok-card",children:[t.jsxs("div",{className:"grok-card-header",children:[t.jsx("div",{className:"grok-card-title",children:"Training Dataset"}),t.jsx(fu,{size:16,className:"text-muted"})]}),t.jsx("input",{ref:N,type:"file",accept:"image/*",multiple:!0,onChange:j=>ee(j.target.files),style:{display:"none"}}),p.length===0?t.jsxs("div",{className:"upload-box",onClick:()=>N.current?.click(),style:{cursor:"pointer"},children:[t.jsx(pt,{size:32,className:"text-muted"}),t.jsx("div",{className:"text-muted",children:"Upload training images (5-20 recommended)"}),t.jsxs("button",{className:"upload-btn",children:[t.jsx(pt,{size:16}),"Select Images"]})]}):t.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"12px"},children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center"},children:[t.jsxs("span",{style:{color:"var(--text-primary)",fontWeight:500},children:[p.length," images selected"]}),t.jsxs("button",{onClick:Z,className:"upload-btn secondary",style:{padding:"4px 8px",fontSize:"0.8rem"},children:[t.jsx(lt,{size:14})," Clear"]})]}),t.jsx("div",{style:{display:"grid",gridTemplateColumns:"repeat(auto-fill, minmax(60px, 1fr))",gap:"8px",maxHeight:"200px",overflowY:"auto",padding:"8px",backgroundColor:"#0f0f0f",borderRadius:"8px",border:"1px solid var(--border-color)"},children:p.map((j,k)=>t.jsx("div",{style:{aspectRatio:"1/1",backgroundColor:"#222",borderRadius:"4px",overflow:"hidden",display:"flex",alignItems:"center",justifyContent:"center"},children:t.jsx("span",{style:{fontSize:"0.6rem",color:"#666"},children:"IMG"})},k))})]})]}),t.jsxs("div",{className:"grok-card",children:[t.jsxs("div",{className:"grok-card-header",children:[t.jsx("div",{className:"grok-card-title",children:"Configuration"}),t.jsx(Yn,{size:16,className:"text-muted"})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Model Name"}),t.jsx("input",{className:"form-input",value:S,onChange:j=>R(j.target.value),placeholder:"e.g. my-style-v1",style:{backgroundColor:"#0f0f0f"}})]}),t.jsxs("div",{className:"form-group",children:[t.jsxs("label",{className:"grok-section-label",children:["Training Epochs (",I,")"]}),t.jsx("input",{type:"range",min:"5",max:"50",step:"5",value:I,onChange:j=>P(parseInt(j.target.value,10)),style:{width:"100%",accentColor:"var(--text-primary)"}}),t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"4px"},children:[t.jsx("span",{children:"Fast (5)"}),t.jsx("span",{children:"Quality (50)"})]})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{className:"grok-section-label",children:"Learning Rate"}),t.jsx("input",{className:"form-input",type:"number",step:"0.00001",value:T,onChange:j=>A(parseFloat(j.target.value||"0")),style:{backgroundColor:"#0f0f0f"}})]})]}),V&&t.jsx("div",{style:{padding:"12px",backgroundColor:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.2)",borderRadius:"8px",color:"#ef4444",marginBottom:"16px",fontSize:"0.9rem"},children:V}),t.jsx("button",{className:"primary-btn",disabled:!D,onClick:K,style:{height:"48px",fontSize:"1rem",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:C?t.jsx(t.Fragment,{children:"Training..."}):t.jsxs(t.Fragment,{children:[t.jsx(Kl,{size:18}),"Start Training"]})})]})}const Ph=[{id:"brief",label:"Brief",description:"1-line summary"},{id:"detailed",label:"Detailed",description:"Full paragraph"},{id:"tags",label:"Tags",description:"Comma-separated keywords"},{id:"structured",label:"Structured",description:"Subject, style, mood"}],Th=[{id:"florence2",label:"Florence-2",description:"Fast & accurate (Microsoft)"},{id:"blip2",label:"BLIP-2",description:"Detailed descriptions"},{id:"cogvlm",label:"CogVLM",description:"High quality (slower)"}];function Rh({onSendToPrompt:c}){const[x,d]=l.useState(null),[N,p]=l.useState(null),[b,S]=l.useState("florence2"),[R,I]=l.useState("detailed"),[P,T]=l.useState(""),[A,C]=l.useState(!1),[L,V]=l.useState(null),U=l.useCallback(j=>{const k=j.target.files?.[0];k&&(d(k),p(URL.createObjectURL(k)),T(""),V(null))},[]),D=l.useCallback(j=>{j.preventDefault();const k=j.dataTransfer.files?.[0];k&&k.type.startsWith("image/")&&(d(k),p(URL.createObjectURL(k)),T(""),V(null))},[]),ee=async()=>{if(x){C(!0),V(null);try{const j=new FormData;j.append("file",x),j.append("model",b),j.append("mode",R);const k=await fetch(`${ve}/caption-image`,{method:"POST",body:j});if(!k.ok){const h=await k.json();throw new Error(h.detail||"Caption failed")}const B=await k.json();T(B.caption||"")}catch(j){console.error("Caption error:",j),V(j.message)}finally{C(!1)}}},Z=()=>{P&&navigator.clipboard.writeText(P)},K=()=>{P&&c&&c(P)};return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(Nn,{size:18}),"Upload Image"]}),t.jsxs("div",{className:`upload-dropzone ${N?"has-preview":""}`,onDrop:D,onDragOver:j=>j.preventDefault(),onClick:()=>document.getElementById("i2t-file-input").click(),children:[N?t.jsx("img",{src:N,alt:"Preview",className:"upload-preview"}):t.jsxs("div",{className:"upload-placeholder",children:[t.jsx(pt,{size:32}),t.jsx("p",{children:"Drop image here or click to upload"})]}),t.jsx("input",{id:"i2t-file-input",type:"file",accept:"image/*",onChange:U,style:{display:"none"}})]})]}),t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(xr,{size:18}),"Caption Settings"]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Model"}),t.jsx("select",{value:b,onChange:j=>S(j.target.value),children:Th.map(j=>t.jsxs("option",{value:j.id,children:[j.label," - ",j.description]},j.id))})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Caption Mode"}),t.jsx("div",{className:"button-group",children:Ph.map(j=>t.jsx("button",{className:`btn-option ${R===j.id?"active":""}`,onClick:()=>I(j.id),title:j.description,children:j.label},j.id))})]})]}),t.jsx("button",{className:"btn-primary btn-large",onClick:ee,disabled:!x||A,children:A?t.jsxs(t.Fragment,{children:[t.jsx(at,{size:18,className:"spin"}),"Generating caption..."]}):t.jsxs(t.Fragment,{children:[t.jsx(xr,{size:18}),"Generate Caption"]})}),L&&t.jsxs("div",{className:"error-message",children:["⚠️ ",L]}),P&&t.jsxs("div",{className:"tool-section result-section",children:[t.jsx("h3",{children:"Generated Caption"}),t.jsxs("div",{className:"caption-result",children:[t.jsx("textarea",{value:P,onChange:j=>T(j.target.value),rows:4}),t.jsxs("div",{className:"caption-actions",children:[t.jsxs("button",{className:"btn-secondary",onClick:Z,children:[t.jsx(un,{size:16}),"Copy"]}),c&&t.jsxs("button",{className:"btn-primary",onClick:K,children:[t.jsx(xu,{size:16}),"Use as Prompt"]})]})]})]}),t.jsx("style",{children:`
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
      `})]})}const Zd=[{id:"cinematic",label:"🎬 Cinematic",keywords:"cinematic lighting, film grain, dramatic shadows, professional photography"},{id:"anime",label:"🎌 Anime",keywords:"anime style, vibrant colors, cel shading, Japanese animation"},{id:"photorealistic",label:"📸 Photorealistic",keywords:"photorealistic, highly detailed, 8k, sharp focus, professional photo"},{id:"abstract",label:"🎨 Abstract",keywords:"abstract art, geometric shapes, vibrant colors, artistic"},{id:"vintage",label:"📼 Vintage",keywords:"vintage aesthetic, retro, film photography, nostalgic, 1970s"},{id:"cyberpunk",label:"🤖 Cyberpunk",keywords:"cyberpunk, neon lights, futuristic, dystopian, high tech low life"},{id:"fantasy",label:"🧙 Fantasy",keywords:"fantasy art, magical, ethereal lighting, mystical, enchanted"},{id:"minimalist",label:"⬜ Minimalist",keywords:"minimalist, clean, simple, negative space, modern"},{id:"horror",label:"👻 Horror",keywords:"dark atmosphere, eerie, horror, unsettling, creepy"},{id:"scifi",label:"🚀 Sci-Fi",keywords:"science fiction, futuristic, space, advanced technology"}];function Mh({onSendToTool:c}){const[x,d]=l.useState(""),[N,p]=l.useState(""),[b,S]=l.useState("expand"),[R,I]=l.useState(!0),[P,T]=l.useState(!1),[A,C]=l.useState(null),[L,V]=l.useState(!1),[U,D]=l.useState(null),ee=async()=>{if(x.trim()){V(!0),D(null);try{const j=await fetch(`${ve}/generate-prompt`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({input:x.trim(),style:N||null,mode:b,include_negative:R,include_motion:P})});if(!j.ok){const B=await j.json();throw new Error(B.detail||"Generation failed")}const k=await j.json();C(k)}catch(j){console.error("Prompt generation error:",j),D(j.message)}finally{V(!1)}}},Z=()=>{if(!x.trim())return;const j=x.trim(),k=Zd.find(re=>re.id===N),B=k?`, ${k.keywords}`:"",h=`${j}${B}, masterpiece, best quality, highly detailed`;C({prompt:h,negative_prompt:R?"ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text, cropped, worst quality":"",motion_prompt:P?"smooth camera motion, cinematic movement, fluid animation":"",variations:null})},K=j=>{navigator.clipboard.writeText(j)};return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(Cn,{size:18}),"Input Idea"]}),t.jsx("textarea",{value:x,onChange:j=>d(j.target.value),placeholder:"Describe your image or video idea... (e.g., 'a cat wearing sunglasses')",rows:3,className:"prompt-input"})]}),t.jsxs("div",{className:"tool-section",children:[t.jsx("h3",{children:"Style Preset"}),t.jsx("div",{className:"style-grid",children:Zd.map(j=>t.jsx("button",{className:`style-btn ${N===j.id?"active":""}`,onClick:()=>p(N===j.id?"":j.id),children:j.label},j.id))})]}),t.jsxs("div",{className:"tool-section",children:[t.jsx("h3",{children:"Options"}),t.jsxs("div",{className:"options-row",children:[t.jsxs("label",{className:"checkbox-label",children:[t.jsx("input",{type:"checkbox",checked:R,onChange:j=>I(j.target.checked)}),"Generate negative prompt"]}),t.jsxs("label",{className:"checkbox-label",children:[t.jsx("input",{type:"checkbox",checked:P,onChange:j=>T(j.target.checked)}),"Include motion prompts (for video)"]})]})]}),t.jsxs("div",{className:"button-row",children:[t.jsxs("button",{className:"btn-primary btn-large",onClick:Z,disabled:!x.trim(),children:[t.jsx(xr,{size:18}),"Quick Generate"]}),t.jsx("button",{className:"btn-secondary btn-large",onClick:ee,disabled:!x.trim()||L,title:"Uses AI for smarter enhancement (requires LLM)",children:L?t.jsxs(t.Fragment,{children:[t.jsx(at,{size:18,className:"spin"}),"Generating..."]}):t.jsxs(t.Fragment,{children:[t.jsx(Cn,{size:18}),"AI Enhance"]})})]}),U&&t.jsxs("div",{className:"error-message",children:["⚠️ ",U]}),A&&t.jsxs("div",{className:"results-section",children:[t.jsxs("div",{className:"result-card",children:[t.jsxs("div",{className:"result-header",children:[t.jsx("h4",{children:"✨ Enhanced Prompt"}),t.jsx("button",{className:"btn-icon",onClick:()=>K(A.prompt),children:t.jsx(un,{size:16})})]}),t.jsx("p",{className:"result-text",children:A.prompt})]}),A.negative_prompt&&t.jsxs("div",{className:"result-card",children:[t.jsxs("div",{className:"result-header",children:[t.jsx("h4",{children:"🚫 Negative Prompt"}),t.jsx("button",{className:"btn-icon",onClick:()=>K(A.negative_prompt),children:t.jsx(un,{size:16})})]}),t.jsx("p",{className:"result-text muted",children:A.negative_prompt})]}),A.motion_prompt&&t.jsxs("div",{className:"result-card",children:[t.jsxs("div",{className:"result-header",children:[t.jsx("h4",{children:"🎬 Motion Prompt"}),t.jsx("button",{className:"btn-icon",onClick:()=>K(A.motion_prompt),children:t.jsx(un,{size:16})})]}),t.jsx("p",{className:"result-text",children:A.motion_prompt})]}),A.variations&&A.variations.length>0&&t.jsxs("div",{className:"result-card",children:[t.jsx("h4",{children:"🔄 Variations"}),A.variations.map((j,k)=>t.jsxs("div",{className:"variation-item",children:[t.jsx("p",{className:"result-text",children:j}),t.jsx("button",{className:"btn-icon",onClick:()=>K(j),children:t.jsx(un,{size:16})})]},k))]}),c&&t.jsxs("button",{className:"btn-primary",onClick:()=>c(A),children:[t.jsx(xu,{size:16}),"Send to Generator"]})]}),t.jsx("style",{children:`
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
      `})]})}const eu=[{value:"CyberRealistic_Pony_v14.1_FP16.safetensors",label:"CyberRealistic Pony"},{value:"dreamshaperXL_lightningDPMSDE.safetensors",label:"Dreamshaper Lightning"},{value:"juggernautXL_ragnarok.safetensors",label:"Juggernaut XL"},{value:"waiIllustriousSDXL_v160.safetensors",label:"Wai Illustrious (Anime)"}];function Lh({onOutput:c,onJobSubmitted:x}){const{user:d,requestLogin:N}=Ke(),[p,b]=l.useState(null),[S,R]=l.useState(null),[I,P]=l.useState(""),[T,A]=l.useState("ugly, deformed, blurry, low quality, bad anatomy, watermark"),[C,L]=l.useState(.6),[V,U]=l.useState("CyberRealistic_Pony_v14.1_FP16.safetensors"),[D,ee]=l.useState(!1),[Z,K]=l.useState(25),[j,k]=l.useState(7),[B,h]=l.useState(-1),[v,te]=l.useState("dpmpp_2m"),[re,xe]=l.useState("karras"),[ge,E]=l.useState(!1),[ue,fe]=l.useState(null),[ie,W]=l.useState(null),[G,X]=l.useState(null),J=l.useCallback(q=>{const le=q.target.files?.[0];le&&(b(le),R(URL.createObjectURL(le)),X(null),fe(null),W(null))},[]),m=l.useCallback(q=>{q.preventDefault();const le=q.dataTransfer.files?.[0];le&&le.type.startsWith("image/")&&(b(le),R(URL.createObjectURL(le)),X(null),fe(null),W(null))},[]),$=async()=>{if(!d){N("Log in om te genereren");return}if(p){E(!0),fe(null),W(null);try{const q=new FormData;q.append("file",p),q.append("prompt",I||"high quality, detailed"),q.append("negative_prompt",T),q.append("denoise",String(C)),q.append("checkpoint",V),q.append("steps",String(Z)),q.append("cfg",String(j)),q.append("seed",String(B)),q.append("sampler_name",v),q.append("scheduler",re);const le=await ht(`${ve}/generate-i2i`,q);if(!le.ok)throw new Error(le.data?.detail||"Generation failed");const F=le.data?.prompt_id;if(!F)throw new Error("No prompt_id returned");W({promptId:F,checkpoint:eu.find(_=>_.value===V)?.label||V}),x&&x({prompt_id:F})}catch(q){console.error("I2I error:",q),fe(q.message)}finally{E(!1)}}};return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(Nn,{size:18}),"Source Image"]}),t.jsxs("div",{className:`upload-dropzone ${S?"has-preview":""}`,onDrop:m,onDragOver:q=>q.preventDefault(),onClick:()=>document.getElementById("i2i-file-input").click(),children:[S?t.jsx("img",{src:S,alt:"Preview",className:"upload-preview"}):t.jsxs("div",{className:"upload-placeholder",children:[t.jsx(pt,{size:32}),t.jsx("p",{children:"Drop image here or click to upload"})]}),t.jsx("input",{id:"i2i-file-input",type:"file",accept:"image/*",onChange:J,style:{display:"none"}})]})]}),t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(xr,{size:18}),"Transformation"]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Prompt (describe desired changes)"}),t.jsx("textarea",{value:I,onChange:q=>P(q.target.value),rows:3,placeholder:"Describe what you want the image to become... (e.g., 'anime style illustration')"})]}),t.jsxs("div",{className:"form-group",children:[t.jsxs("label",{children:[t.jsx(zs,{size:14}),"Denoise Strength",t.jsx("span",{className:"label-value",children:C.toFixed(2)})]}),t.jsx("input",{type:"range",min:"0.1",max:"1.0",step:"0.05",value:C,onChange:q=>L(parseFloat(q.target.value))}),t.jsxs("div",{className:"range-labels",children:[t.jsx("span",{children:"Subtle (0.1)"}),t.jsx("span",{children:"Complete (1.0)"})]}),t.jsxs("div",{className:"denoise-hint",children:[C<.3&&"💡 Minor adjustments, preserves most of original",C>=.3&&C<.6&&"💡 Moderate changes, good balance",C>=.6&&C<.8&&"💡 Significant transformation",C>=.8&&"💡 Near-complete regeneration from prompt"]})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Model"}),t.jsx("select",{value:V,onChange:q=>U(q.target.value),children:eu.map(q=>t.jsx("option",{value:q.value,children:q.label},q.value))})]})]}),t.jsxs("div",{className:"tool-section collapsible",children:[t.jsxs("button",{className:"section-toggle",onClick:()=>ee(!D),children:[t.jsx(Yn,{size:16}),"Advanced Settings",t.jsx(Qt,{size:16,className:D?"rotated":""})]}),D&&t.jsxs("div",{className:"advanced-content",children:[t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Negative Prompt"}),t.jsx("textarea",{value:T,onChange:q=>A(q.target.value),rows:2})]}),t.jsxs("div",{className:"form-row",children:[t.jsxs("div",{className:"form-group half",children:[t.jsx("label",{children:"Steps"}),t.jsx("input",{type:"number",value:Z,onChange:q=>K(parseInt(q.target.value)||25),min:"1",max:"50"})]}),t.jsxs("div",{className:"form-group half",children:[t.jsx("label",{children:"CFG Scale"}),t.jsx("input",{type:"number",value:j,onChange:q=>k(parseFloat(q.target.value)||7),min:"1",max:"20",step:"0.5"})]})]}),t.jsxs("div",{className:"form-row",children:[t.jsxs("div",{className:"form-group half",children:[t.jsx("label",{children:"Sampler"}),t.jsxs("select",{value:v,onChange:q=>te(q.target.value),children:[t.jsx("option",{value:"euler",children:"Euler"}),t.jsx("option",{value:"euler_ancestral",children:"Euler Ancestral"}),t.jsx("option",{value:"dpmpp_2m",children:"DPM++ 2M"}),t.jsx("option",{value:"dpmpp_2m_sde",children:"DPM++ 2M SDE"}),t.jsx("option",{value:"dpmpp_3m_sde",children:"DPM++ 3M SDE"})]})]}),t.jsxs("div",{className:"form-group half",children:[t.jsx("label",{children:"Scheduler"}),t.jsxs("select",{value:re,onChange:q=>xe(q.target.value),children:[t.jsx("option",{value:"normal",children:"Normal"}),t.jsx("option",{value:"karras",children:"Karras"}),t.jsx("option",{value:"exponential",children:"Exponential"}),t.jsx("option",{value:"sgm_uniform",children:"SGM Uniform"})]})]})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Seed (-1 = random)"}),t.jsx("input",{type:"number",value:B,onChange:q=>h(parseInt(q.target.value)||-1)})]})]})]}),ie&&t.jsxs("div",{className:"queued-notice",children:["✅ Job queued! Check the Queue panel for progress.",t.jsx("span",{className:"queued-mode",children:ie.checkpoint})]}),ue&&t.jsxs("div",{className:"error-message",children:["⚠️ ",ue]}),t.jsx("button",{className:"btn-primary btn-large",onClick:$,disabled:!p||ge,children:ge?t.jsxs(t.Fragment,{children:[t.jsx(at,{size:18,className:"spin"}),"Queueing..."]}):t.jsxs(t.Fragment,{children:[t.jsx(xr,{size:18}),"Transform Image"]})}),G&&t.jsxs("div",{className:"result-section",children:[t.jsx("h3",{children:"Result"}),t.jsxs("div",{className:"comparison",children:[t.jsxs("div",{className:"comparison-item",children:[t.jsx("span",{className:"comparison-label",children:"Original"}),t.jsx("img",{src:S,alt:"Original"})]}),t.jsxs("div",{className:"comparison-item",children:[t.jsx("span",{className:"comparison-label",children:"Transformed"}),t.jsx("img",{src:G,alt:"Result"})]})]})]}),t.jsx("style",{children:`
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
      `})]})}const Ol=[{value:"RealESRGAN_x4plus.pth",label:"RealESRGAN 4x (General)",scale:4},{value:"RealESRGAN_x4plus_anime_6B.pth",label:"RealESRGAN 4x (Anime)",scale:4},{value:"RealESRGAN_x2plus.pth",label:"RealESRGAN 2x",scale:2},{value:"4x-UltraSharp.pth",label:"4x UltraSharp",scale:4},{value:"4x_NMKD-Siax_200k.pth",label:"4x NMKD-Siax",scale:4}],Fh=[2,4];function Dh({onOutput:c,onJobSubmitted:x}){const{user:d,requestLogin:N}=Ke(),[p,b]=l.useState(null),[S,R]=l.useState(null),[I,P]=l.useState(null),[T,A]=l.useState("RealESRGAN_x4plus.pth"),[C,L]=l.useState(4),[V,U]=l.useState(!1),[D,ee]=l.useState(!1),[Z,K]=l.useState(null),[j,k]=l.useState(null),[B,h]=l.useState(null),v=l.useCallback(E=>{const ue=E.target.files?.[0];if(ue){b(ue);const fe=URL.createObjectURL(ue);R(fe),h(null),K(null),k(null);const ie=new Image;ie.onload=()=>{P({width:ie.width,height:ie.height})},ie.src=fe}},[]),te=l.useCallback(E=>{E.preventDefault();const ue=E.dataTransfer.files?.[0];if(ue&&ue.type.startsWith("image/")){b(ue);const fe=URL.createObjectURL(ue);R(fe),h(null),K(null),k(null);const ie=new Image;ie.onload=()=>{P({width:ie.width,height:ie.height})},ie.src=fe}},[]),re=async()=>{if(!d){N("Log in om te genereren");return}if(p){ee(!0),K(null),k(null);try{const E=new FormData;E.append("file",p),E.append("model",T),E.append("scale",String(C)),E.append("face_enhance",String(V));const ue=await ht(`${ve}/upscale`,E);if(!ue.ok)throw new Error(ue.data?.detail||"Upscaling failed");const fe=ue.data?.prompt_id;if(!fe)throw new Error("No prompt_id returned");k({promptId:fe,model:Ol.find(ie=>ie.value===T)?.label||T,scale:C}),x&&x({prompt_id:fe})}catch(E){console.error("Upscale error:",E),K(E.message)}finally{ee(!1)}}};Ol.find(E=>E.value===T);const xe=I?I.width*C:0,ge=I?I.height*C:0;return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(Nn,{size:18}),"Source Image"]}),t.jsxs("div",{className:`upload-dropzone ${S?"has-preview":""}`,onDrop:te,onDragOver:E=>E.preventDefault(),onClick:()=>document.getElementById("upscale-file-input").click(),children:[S?t.jsx("img",{src:S,alt:"Preview",className:"upload-preview"}):t.jsxs("div",{className:"upload-placeholder",children:[t.jsx(pt,{size:32}),t.jsx("p",{children:"Drop image here or click to upload"})]}),t.jsx("input",{id:"upscale-file-input",type:"file",accept:"image/*",onChange:v,style:{display:"none"}})]}),I&&t.jsxs("div",{className:"image-info",children:[t.jsxs("span",{children:["📐 ",I.width," × ",I.height,"px"]}),t.jsx("span",{children:"→"}),t.jsxs("span",{className:"output-size",children:[xe," × ",ge,"px"]})]})]}),t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(Hl,{size:18}),"Upscale Settings"]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Scale Factor"}),t.jsx("div",{className:"button-group",children:Fh.map(E=>t.jsxs("button",{className:`btn-option ${C===E?"active":""}`,onClick:()=>L(E),type:"button",children:[E,"x"]},E))})]}),t.jsxs("div",{className:"form-group",children:[t.jsx("label",{children:"Upscale Model"}),t.jsx("select",{value:T,onChange:E=>A(E.target.value),children:Ol.map(E=>t.jsx("option",{value:E.value,children:E.label},E.value))})]}),t.jsx("div",{className:"form-group",children:t.jsxs("label",{className:"checkbox-label",children:[t.jsx("input",{type:"checkbox",checked:V,onChange:E=>U(E.target.checked)}),"Face Enhancement (GFPGAN)",t.jsx("span",{className:"hint",children:"Improves face details"})]})})]}),j&&t.jsxs("div",{className:"queued-notice",children:["✅ Job queued! Check the Queue panel for progress.",t.jsxs("span",{className:"queued-mode",children:[j.scale,"x ",j.model]})]}),Z&&t.jsxs("div",{className:"error-message",children:["⚠️ ",Z]}),t.jsx("button",{className:"btn-primary btn-large",onClick:re,disabled:!p||D,children:D?t.jsxs(t.Fragment,{children:[t.jsx(at,{size:18,className:"spin"}),"Queueing..."]}):t.jsxs(t.Fragment,{children:[t.jsx(Hl,{size:18}),"Upscale Image"]})}),B&&t.jsxs("div",{className:"result-section",children:[t.jsxs("h3",{children:["Result (",C,"x Upscaled)"]}),t.jsx("div",{className:"result-image",children:t.jsx("img",{src:B,alt:"Upscaled"})}),t.jsx("a",{href:B,download:!0,className:"btn-secondary",style:{marginTop:12,display:"inline-flex",alignItems:"center",gap:8},children:"Download Full Resolution"})]}),t.jsx("style",{children:`
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
      `})]})}const tu=[{value:"nova",label:"Nova",desc:"Friendly, upbeat",gender:"female"},{value:"shimmer",label:"Shimmer",desc:"Soft, gentle",gender:"female"},{value:"alloy",label:"Alloy",desc:"Neutral, versatile",gender:"female"},{value:"echo",label:"Echo",desc:"Warm, conversational",gender:"male"},{value:"fable",label:"Fable",desc:"Expressive, dramatic",gender:"male"},{value:"onyx",label:"Onyx",desc:"Deep, authoritative",gender:"male"}],Oh=[{value:"tts",label:"Text to Speech",icon:t.jsx(Xl,{size:18}),desc:"Generate voice from text"},{value:"music",label:"Music Generation",icon:t.jsx(Am,{size:18}),desc:"Generate music/sounds"},{value:"sfx",label:"Sound Effects",icon:t.jsx(Br,{size:18}),desc:"Generate sound effects"}],Ah=[{value:"ambient",label:"Ambient"},{value:"cinematic",label:"Cinematic"},{value:"electronic",label:"Electronic"},{value:"jazz",label:"Jazz"},{value:"classical",label:"Classical"},{value:"lofi",label:"Lo-Fi"},{value:"rock",label:"Rock"},{value:"hiphop",label:"Hip-Hop"}];function $h({onOutput:c,onJobSubmitted:x}){const{user:d,requestLogin:N}=Ke(),[p,b]=l.useState("tts"),[S,R]=l.useState(""),[I,P]=l.useState("nova"),[T,A]=l.useState("cinematic"),[C,L]=l.useState(10),[V,U]=l.useState(!1),[D,ee]=l.useState(1),[Z,K]=l.useState(1),[j,k]=l.useState(!1),[B,h]=l.useState(null),[v,te]=l.useState(null),[re,xe]=l.useState(null),[ge,E]=l.useState(!1),ue=l.useRef(null),fe=async()=>{if(!d){N("Log in om te genereren");return}if(S.trim()){k(!0),h(null),te(null);try{let W="/generate-audio";const G=new FormData;G.append("text",S.trim()),G.append("mode",p),p==="tts"?(G.append("voice",I),G.append("speed",D.toString()),G.append("pitch",Z.toString())):p==="music"?(G.append("style",T),G.append("duration",C.toString())):p==="sfx"&&G.append("duration",Math.min(C,10).toString());const X=await ht(`${ve}${W}`,G);if(!X.ok){const J=typeof X.data=="object"?X.data?.detail||JSON.stringify(X.data):X.data||"Audio generation failed";throw new Error(J)}if(X.data?.prompt_id)te({promptId:X.data.prompt_id,mode:p,text:S.substring(0,50)+(S.length>50?"...":"")}),x&&x(X.data);else if(X.data?.url){const J=X.data.url,m=J.startsWith("http")?J:`${ve}${J}`;xe({url:m,filename:J.split("/").pop()}),c&&c({kind:"audio",url:m,filename:J.split("/").pop()})}}catch(W){console.error("Audio error:",W),h(W.message)}finally{k(!1)}}},ie=()=>{ue.current&&(ge?ue.current.pause():ue.current.play(),E(!ge))};return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(Br,{size:18}),"Generation Mode"]}),t.jsx("div",{className:"mode-grid",children:Oh.map(W=>t.jsxs("button",{className:`mode-btn ${p===W.value?"active":""}`,onClick:()=>b(W.value),children:[W.icon,t.jsx("span",{className:"mode-name",children:W.label}),t.jsx("span",{className:"mode-desc",children:W.desc})]},W.value))})]}),t.jsxs("div",{className:"tool-section",children:[t.jsx("h3",{children:p==="tts"?"Text to Speak":p==="music"?"Music Prompt":"Sound Description"}),t.jsx("textarea",{value:S,onChange:W=>R(W.target.value),placeholder:p==="tts"?"Enter the text you want to convert to speech...":p==="music"?'Describe the music you want to generate (e.g., "upbeat electronic dance track with heavy bass")':'Describe the sound effect (e.g., "thunder rumbling in the distance")',rows:4,className:"prompt-textarea"})]}),p==="tts"&&t.jsxs("div",{className:"tool-section",children:[t.jsx("h3",{children:"Voice"}),t.jsxs("div",{className:"voice-group",children:[t.jsx("span",{className:"voice-group-label",children:"Female"}),t.jsx("div",{className:"voice-grid",children:tu.filter(W=>W.gender==="female").map(W=>t.jsxs("button",{className:`voice-btn ${I===W.value?"active":""}`,onClick:()=>P(W.value),children:[t.jsx("span",{className:"voice-name",children:W.label}),t.jsx("span",{className:"voice-desc",children:W.desc})]},W.value))})]}),t.jsxs("div",{className:"voice-group",children:[t.jsx("span",{className:"voice-group-label",children:"Male"}),t.jsx("div",{className:"voice-grid",children:tu.filter(W=>W.gender==="male").map(W=>t.jsxs("button",{className:`voice-btn ${I===W.value?"active":""}`,onClick:()=>P(W.value),children:[t.jsx("span",{className:"voice-name",children:W.label}),t.jsx("span",{className:"voice-desc",children:W.desc})]},W.value))})]})]}),p==="music"&&t.jsxs("div",{className:"tool-section",children:[t.jsx("h3",{children:"Style"}),t.jsx("div",{className:"style-grid",children:Ah.map(W=>t.jsx("button",{className:`style-btn ${T===W.value?"active":""}`,onClick:()=>A(W.value),children:W.label},W.value))})]}),(p==="music"||p==="sfx")&&t.jsxs("div",{className:"tool-section",children:[t.jsx("h3",{children:"Duration"}),t.jsxs("div",{className:"slider-row",children:[t.jsx("input",{type:"range",min:p==="sfx"?1:5,max:p==="sfx"?10:30,value:C,onChange:W=>L(parseInt(W.target.value))}),t.jsxs("span",{className:"slider-value",children:[C,"s"]})]})]}),p==="tts"&&t.jsxs("div",{className:"tool-section collapsible",children:[t.jsxs("h3",{onClick:()=>U(!V),style:{cursor:"pointer"},children:[t.jsx(Yn,{size:16}),"Advanced",t.jsx(Qt,{size:16,style:{marginLeft:"auto",transform:V?"rotate(180deg)":"rotate(0deg)",transition:"transform 0.2s"}})]}),V&&t.jsxs("div",{className:"advanced-content",children:[t.jsxs("div",{className:"slider-row",children:[t.jsx("label",{children:"Speed"}),t.jsx("input",{type:"range",min:.5,max:2,step:.1,value:D,onChange:W=>ee(parseFloat(W.target.value))}),t.jsxs("span",{className:"slider-value",children:[D.toFixed(1),"x"]})]}),t.jsxs("div",{className:"slider-row",children:[t.jsx("label",{children:"Pitch"}),t.jsx("input",{type:"range",min:.5,max:2,step:.1,value:Z,onChange:W=>K(parseFloat(W.target.value))}),t.jsxs("span",{className:"slider-value",children:[Z.toFixed(1),"x"]})]})]})]}),v&&t.jsxs("div",{className:"queued-notice",children:["✅ Job queued! Check the Queue panel for progress.",t.jsx("span",{className:"queued-mode",children:v.mode.toUpperCase()})]}),B&&t.jsxs("div",{className:"error-message",children:["⚠️ ",B]}),t.jsx("button",{className:"btn-primary btn-large",onClick:fe,disabled:!S.trim()||j,children:j?t.jsxs(t.Fragment,{children:[t.jsx(at,{size:18,className:"spin"}),"Queueing..."]}):t.jsxs(t.Fragment,{children:[t.jsx(Br,{size:18}),"Generate ",p==="tts"?"Speech":p==="music"?"Music":"Sound"]})}),re&&t.jsxs("div",{className:"result-section",children:[t.jsx("h3",{children:"Result"}),t.jsxs("div",{className:"audio-player",children:[t.jsx("audio",{ref:ue,src:re.url,onEnded:()=>E(!1),onPlay:()=>E(!0),onPause:()=>E(!1)}),t.jsx("button",{className:"play-btn",onClick:ie,children:ge?t.jsx(Wl,{size:24}):t.jsx(Va,{size:24})}),t.jsx("div",{className:"audio-info",children:t.jsx("span",{className:"audio-filename",children:re.filename})}),t.jsx("a",{href:re.url,download:!0,className:"download-btn",children:t.jsx(qt,{size:18})})]})]}),t.jsx("style",{children:`
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
      `})]})}const Uh=["audio/wav","audio/mp3","audio/mpeg","audio/flac","audio/ogg","audio/webm"],nu=[{value:"F5v1",label:"F5 v1 (English)",desc:"Best quality English"},{value:"F5",label:"F5 Base (English)",desc:"Standard English model"},{value:"F5-DE",label:"F5 German",desc:"German language"},{value:"F5-FR",label:"F5 French",desc:"French language"},{value:"F5-ES",label:"F5 Spanish",desc:"Spanish language"},{value:"F5-IT",label:"F5 Italian",desc:"Italian language"},{value:"F5-JP",label:"F5 Japanese",desc:"Japanese language"},{value:"E2",label:"E2-TTS",desc:"Alternative English model"}];function Vh({onOutput:c,onJobSubmitted:x}){const{user:d,requestLogin:N}=Ke(),[p,b]=l.useState(null),[S,R]=l.useState(null),[I,P]=l.useState(null),[T,A]=l.useState(""),[C,L]=l.useState("F5v1"),[V,U]=l.useState(1),[D,ee]=l.useState(!1),[Z,K]=l.useState(0),j=l.useRef(null),k=l.useRef([]),B=l.useRef(null),h=l.useRef(null),v=l.useRef(null),[te,re]=l.useState(!1),[xe,ge]=l.useState(!1),[E,ue]=l.useState(!1),[fe,ie]=l.useState(!1),[W,G]=l.useState(null),[X,J]=l.useState(null),[m,$]=l.useState(null),q=l.useCallback(pe=>{pe.preventDefault();const Ne=pe.dataTransfer?.files?.[0]||pe.target?.files?.[0];Ne&&Uh.some(Re=>Ne.type.includes(Re.split("/")[1]))?(b(Ne),R(URL.createObjectURL(Ne)),P(null),G(null)):Ne&&G("Please upload a valid audio file (WAV, MP3, FLAC, OGG)")},[]),le=async()=>{try{const pe=await navigator.mediaDevices.getUserMedia({audio:!0}),Ne=new MediaRecorder(pe,{mimeType:"audio/webm;codecs=opus"});k.current=[],j.current=Ne,Ne.ondataavailable=Re=>{Re.data.size>0&&k.current.push(Re.data)},Ne.onstop=()=>{const Re=new Blob(k.current,{type:"audio/webm"}),Ve=new File([Re],"recording.webm",{type:"audio/webm"});b(Ve),R(URL.createObjectURL(Re)),P(null),pe.getTracks().forEach(it=>it.stop())},Ne.start(),ee(!0),K(0),B.current=setInterval(()=>{K(Re=>Re+1)},1e3)}catch(pe){G("Failed to access microphone: "+pe.message)}},F=()=>{j.current&&D&&(j.current.stop(),ee(!1),clearInterval(B.current))},_=async()=>{if(!p)return null;const pe=new FormData;pe.append("file",p);try{const Ne=await ht(`${ve}/upload`,pe);if(Ne.ok&&Ne.data?.path)return P(Ne.data.path),Ne.data.path;throw new Error(Ne.data?.detail||"Upload failed")}catch(Ne){throw new Error("Failed to upload voice sample: "+Ne.message)}},Y=async()=>{if(!d){N("Log in om te genereren");return}if(!p||!T.trim()){G("Please provide both a voice sample and text to speak");return}ue(!0),ie(!0),G(null),J(null),$(null);try{let pe=I;pe||(pe=await _()),ie(!1);const Ne=await Es(`${ve}/voice-clone`,{voice_sample_path:pe,text:T.trim(),model:C,speed:V});if(!Ne.ok)throw new Error(Ne.data?.detail||"Voice cloning request failed");Ne.data?.prompt_id&&(J({promptId:Ne.data.prompt_id,model:nu.find(Re=>Re.value===C)?.label||C}),x&&x({prompt_id:Ne.data.prompt_id}))}catch(pe){console.error("Voice cloning error:",pe),G(pe.message)}finally{ue(!1),ie(!1)}},Q=()=>{b(null),R(null),P(null),J(null),h.current&&(h.current.pause(),h.current.currentTime=0),re(!1)},u=()=>{h.current&&(te?h.current.pause():h.current.play(),re(!te))},he=()=>{v.current&&(xe?v.current.pause():v.current.play(),ge(!xe))},ze=pe=>{const Ne=Math.floor(pe/60),Re=pe%60;return`${Ne}:${Re.toString().padStart(2,"0")}`};return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(Vl,{size:18}),"Voice Sample (5-30 seconds recommended)"]}),p?t.jsxs("div",{className:"voice-preview",children:[t.jsxs("div",{className:"voice-file-info",children:[t.jsx(Vl,{size:24}),t.jsxs("div",{className:"file-details",children:[t.jsx("span",{className:"filename",children:p.name}),t.jsxs("span",{className:"filesize",children:[(p.size/1024).toFixed(1)," KB"]})]}),t.jsxs("div",{className:"voice-controls",children:[t.jsx("button",{className:"icon-btn",onClick:u,children:te?t.jsx(Wl,{size:18}):t.jsx(Va,{size:18})}),t.jsx("button",{className:"icon-btn danger",onClick:Q,children:t.jsx(Ba,{size:18})})]})]}),t.jsx("audio",{ref:h,src:S,onEnded:()=>re(!1)}),I&&t.jsxs("div",{className:"upload-status",children:[t.jsx(mr,{size:14})," Uploaded"]})]}):t.jsxs("div",{className:"voice-input-options",children:[t.jsxs("div",{className:"drop-zone",onDrop:q,onDragOver:pe=>pe.preventDefault(),onClick:()=>document.getElementById("voice-file-input").click(),children:[t.jsx(pt,{size:32}),t.jsx("p",{children:"Drop audio file here or click to browse"}),t.jsx("span",{className:"supported-formats",children:"WAV, MP3, FLAC, OGG"}),t.jsx("input",{id:"voice-file-input",type:"file",accept:"audio/*",onChange:q,style:{display:"none"}})]}),t.jsx("div",{className:"divider-text",children:"or"}),t.jsx("button",{className:`record-btn ${D?"recording":""}`,onClick:D?F:le,children:D?t.jsxs(t.Fragment,{children:[t.jsx("div",{className:"recording-indicator"}),t.jsxs("span",{children:["Stop Recording (",ze(Z),")"]})]}):t.jsxs(t.Fragment,{children:[t.jsx(Xl,{size:20}),t.jsx("span",{children:"Record Voice Sample"})]})})]})]}),t.jsxs("div",{className:"tool-section",children:[t.jsx("h3",{children:"Text to Speak"}),t.jsx("textarea",{value:T,onChange:pe=>A(pe.target.value),placeholder:"Enter the text you want the cloned voice to speak...",rows:4,className:"prompt-textarea"}),t.jsxs("div",{className:"char-count",children:[T.length," characters"]})]}),t.jsxs("div",{className:"tool-section",children:[t.jsx("h3",{children:"Model"}),t.jsx("div",{className:"model-grid",children:nu.map(pe=>t.jsxs("button",{className:`model-btn ${C===pe.value?"active":""}`,onClick:()=>L(pe.value),children:[t.jsx("span",{className:"model-name",children:pe.label}),t.jsx("span",{className:"model-desc",children:pe.desc})]},pe.value))})]}),t.jsxs("div",{className:"tool-section",children:[t.jsx("h3",{children:"Speed"}),t.jsxs("div",{className:"slider-row",children:[t.jsx("input",{type:"range",min:.5,max:2,step:.1,value:V,onChange:pe=>U(parseFloat(pe.target.value))}),t.jsxs("span",{className:"slider-value",children:[V.toFixed(1),"x"]})]}),t.jsxs("div",{className:"slider-hints",children:[t.jsx("span",{children:">1.0 = slower"}),t.jsx("span",{children:"<1.0 = faster"})]})]}),X&&t.jsxs("div",{className:"queued-notice",children:["✅ Job queued! Check the Queue panel for progress.",t.jsx("span",{className:"queued-mode",children:X.model})]}),t.jsx("div",{className:"tool-section",children:t.jsx("button",{className:"generate-btn",onClick:Y,disabled:E||!p||!T.trim(),children:E?t.jsxs(t.Fragment,{children:[t.jsx(at,{size:20,className:"spin"}),t.jsx("span",{children:fe?"Uploading...":"Queueing..."})]}):t.jsxs(t.Fragment,{children:[t.jsx(Br,{size:20}),t.jsx("span",{children:"Clone Voice"})]})})}),W&&t.jsxs("div",{className:"error-message",children:[t.jsx(lt,{size:16}),W]}),m&&t.jsxs("div",{className:"tool-section result-section",children:[t.jsxs("h3",{children:[t.jsx(Br,{size:18}),"Cloned Voice Result"]}),t.jsxs("div",{className:"audio-result",children:[t.jsx("audio",{ref:v,src:m.url,onEnded:()=>ge(!1)}),t.jsxs("div",{className:"audio-controls",children:[t.jsx("button",{className:"play-btn",onClick:he,children:xe?t.jsx(Wl,{size:24}):t.jsx(Va,{size:24})}),t.jsx("span",{className:"filename",children:m.filename}),t.jsx("a",{href:m.url,download:m.filename,className:"download-btn",children:t.jsx(qt,{size:18})})]})]})]}),t.jsx("style",{children:`
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
      `})]})}const Bh=["video/mp4","video/webm","video/quicktime"],Wh=["audio/wav","audio/mp3","audio/mpeg","audio/flac","audio/ogg","audio/webm"];function Gh({onOutput:c,onJobSubmitted:x}){const{user:d,requestLogin:N}=Ke(),[p,b]=l.useState(null),[S,R]=l.useState(null),[I,P]=l.useState(null),[T,A]=l.useState(null),[C,L]=l.useState(null),[V,U]=l.useState(null),[D,ee]=l.useState(1.5),[Z,K]=l.useState(20),[j,k]=l.useState(-1),B=l.useRef(null),h=l.useRef(null),v=l.useRef(null),[te,re]=l.useState(!1),[xe,ge]=l.useState(!1),[E,ue]=l.useState(null),[fe,ie]=l.useState(null),[W,G]=l.useState(null),X=l.useCallback(F=>{F.preventDefault();const _=F.dataTransfer?.files?.[0]||F.target?.files?.[0];_&&Bh.some(Y=>_.type.includes(Y.split("/")[1]))?(b(_),R(URL.createObjectURL(_)),P(null),ue(null),ie(null)):_&&ue("Please upload a valid video file (MP4, WebM)")},[]),J=l.useCallback(F=>{F.preventDefault();const _=F.dataTransfer?.files?.[0]||F.target?.files?.[0];_&&Wh.some(Y=>_.type.includes(Y.split("/")[1]))?(A(_),L(URL.createObjectURL(_)),U(null),ue(null),ie(null)):_&&ue("Please upload a valid audio file (WAV, MP3, FLAC)")},[]),m=async F=>{const _=new FormData;_.append("file",F);try{const Y=await ht(`${ve}/upload`,_);if(Y.ok&&Y.data?.path)return Y.data.path;throw new Error(Y.data?.detail||"Upload failed")}catch(Y){throw new Error("Failed to upload file: "+Y.message)}},$=async()=>{if(!d){N("Log in om te genereren");return}if(!p||!T){ue("Please provide both a video and audio file");return}re(!0),ge(!0),ue(null),ie(null),G(null);try{let F=I;F||(F=await m(p),P(F));let _=V;_||(_=await m(T),U(_)),ge(!1);const Y=await Es(`${ve}/lip-sync`,{video_path:F,audio_path:_,lips_expression:D,inference_steps:Z,seed:j===-1?Math.floor(Math.random()*2147483647):j});if(!Y.ok)throw new Error(Y.data?.detail||"Lip sync request failed");Y.data?.prompt_id&&(ie({promptId:Y.data.prompt_id}),x&&x({prompt_id:Y.data.prompt_id}))}catch(F){console.error("Lip sync error:",F),ue(F.message)}finally{re(!1),ge(!1)}},q=()=>{b(null),R(null),P(null)},le=()=>{A(null),L(null),U(null)};return t.jsxs("div",{className:"tool-container",children:[t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(dm,{size:18}),"Input Video (with face)"]}),p?t.jsxs("div",{className:"media-preview",children:[t.jsx("video",{ref:B,src:S,controls:!0,className:"preview-video"}),t.jsxs("div",{className:"file-info-row",children:[t.jsx("span",{className:"filename",children:p.name}),t.jsx("button",{className:"icon-btn danger",onClick:q,children:t.jsx(Ba,{size:18})})]})]}):t.jsxs("div",{className:"drop-zone",onDrop:X,onDragOver:F=>F.preventDefault(),onClick:()=>document.getElementById("video-file-input").click(),children:[t.jsx(pt,{size:32}),t.jsx("p",{children:"Drop video file here or click to browse"}),t.jsx("span",{className:"supported-formats",children:"MP4, WebM"}),t.jsx("input",{id:"video-file-input",type:"file",accept:"video/*",onChange:X,style:{display:"none"}})]})]}),t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(Vl,{size:18}),"Audio Track (speech/dialogue)"]}),T?t.jsxs("div",{className:"audio-preview",children:[t.jsx("audio",{ref:h,src:C,controls:!0,className:"preview-audio"}),t.jsxs("div",{className:"file-info-row",children:[t.jsx("span",{className:"filename",children:T.name}),t.jsx("button",{className:"icon-btn danger",onClick:le,children:t.jsx(Ba,{size:18})})]})]}):t.jsxs("div",{className:"drop-zone",onDrop:J,onDragOver:F=>F.preventDefault(),onClick:()=>document.getElementById("audio-file-input").click(),children:[t.jsx(pt,{size:32}),t.jsx("p",{children:"Drop audio file here or click to browse"}),t.jsx("span",{className:"supported-formats",children:"WAV, MP3, FLAC, OGG"}),t.jsx("input",{id:"audio-file-input",type:"file",accept:"audio/*",onChange:J,style:{display:"none"}})]})]}),t.jsxs("div",{className:"tool-section",children:[t.jsxs("h3",{children:[t.jsx(zs,{size:18}),"Settings"]}),t.jsxs("div",{className:"setting-row",children:[t.jsx("label",{children:"Lips Expression"}),t.jsxs("div",{className:"slider-row",children:[t.jsx("input",{type:"range",min:1,max:3,step:.1,value:D,onChange:F=>ee(parseFloat(F.target.value))}),t.jsx("span",{className:"slider-value",children:D.toFixed(1)})]}),t.jsx("span",{className:"setting-hint",children:"Higher = more exaggerated lip movements"})]}),t.jsxs("div",{className:"setting-row",children:[t.jsx("label",{children:"Inference Steps"}),t.jsxs("div",{className:"slider-row",children:[t.jsx("input",{type:"range",min:10,max:50,step:5,value:Z,onChange:F=>K(parseInt(F.target.value))}),t.jsx("span",{className:"slider-value",children:Z})]}),t.jsx("span",{className:"setting-hint",children:"More steps = better quality, slower"})]}),t.jsxs("div",{className:"setting-row",children:[t.jsx("label",{children:"Seed"}),t.jsx("input",{type:"number",value:j,onChange:F=>k(parseInt(F.target.value)||-1),placeholder:"-1 for random",className:"seed-input"})]})]}),fe&&t.jsx("div",{className:"queued-notice",children:"✅ Job queued! Check the Queue panel for progress."}),t.jsx("div",{className:"tool-section",children:t.jsx("button",{className:"generate-btn",onClick:$,disabled:te||!p||!T,children:te?t.jsxs(t.Fragment,{children:[t.jsx(at,{size:20,className:"spin"}),t.jsx("span",{children:xe?"Uploading...":"Queueing..."})]}):t.jsxs(t.Fragment,{children:[t.jsx(Xn,{size:20}),t.jsx("span",{children:"Sync Lips"})]})})}),E&&t.jsxs("div",{className:"error-message",children:[t.jsx(lt,{size:16}),E]}),W&&t.jsxs("div",{className:"tool-section result-section",children:[t.jsxs("h3",{children:[t.jsx(Xn,{size:18}),"Lip Synced Result"]}),t.jsxs("div",{className:"video-result",children:[t.jsx("video",{ref:v,src:W.url,controls:!0,className:"result-video"}),t.jsxs("div",{className:"result-actions",children:[t.jsx("span",{className:"filename",children:W.filename}),t.jsxs("a",{href:W.url,download:W.filename,className:"download-btn",children:[t.jsx(qt,{size:18}),"Download"]})]})]})]}),t.jsx("style",{children:`
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
      `})]})}const ru=[{id:"1:1",label:"1:1 (Square)",width:1024,height:1024},{id:"16:9",label:"16:9 (Widescreen)",width:1280,height:720},{id:"9:16",label:"9:16 (Portrait)",width:720,height:1280},{id:"4:3",label:"4:3 (Standard)",width:1024,height:768},{id:"3:4",label:"3:4 (Portrait)",width:768,height:1024},{id:"21:9",label:"21:9 (Ultrawide)",width:1344,height:576},{id:"3:2",label:"3:2 (Photo)",width:1152,height:768},{id:"2:3",label:"2:3 (Photo Portrait)",width:768,height:1152}],Hh=[{id:"center",label:"Center",icon:"⊕"},{id:"top",label:"Top",icon:"⬆️"},{id:"bottom",label:"Bottom",icon:"⬇️"},{id:"left",label:"Left",icon:"⬅️"},{id:"right",label:"Right",icon:"➡️"},{id:"top-left",label:"Top Left",icon:"↖️"},{id:"top-right",label:"Top Right",icon:"↗️"},{id:"bottom-left",label:"Bottom Left",icon:"↙️"},{id:"bottom-right",label:"Bottom Right",icon:"↘️"}],su=[{id:"sdxl",label:"SDXL (Quality)",file:"CyberRealisticPony_v8.safetensors"},{id:"flux",label:"Flux (Fast)",file:"flux1-dev-bnb-nf4.safetensors"}];function qh({onJobSubmitted:c}){const{user:x,requestLogin:d}=Ke(),[N,p]=l.useState(null),[b,S]=l.useState(null),[R,I]=l.useState({width:0,height:0}),[P,T]=l.useState(ru[0]),[A,C]=l.useState("center"),[L,V]=l.useState(su[0]),[U,D]=l.useState(""),[ee,Z]=l.useState(25),[K,j]=l.useState(7),[k,B]=l.useState(.85),[h,v]=l.useState(32),[te,re]=l.useState(!1),[xe,ge]=l.useState(null),[E,ue]=l.useState(null),[fe,ie]=l.useState(!1),[W,G]=l.useState(null),X=l.useRef(null),J=l.useCallback(_=>{_.preventDefault();const Y=_.dataTransfer?.files?.[0]||_.target?.files?.[0];if(Y&&Y.type.startsWith("image/")){p(Y),ge(null),ue(null),G(null);const Q=URL.createObjectURL(Y),u=new Image;u.onload=()=>{I({width:u.naturalWidth,height:u.naturalHeight}),S(Q)},u.src=Q}},[]),m=_=>_.preventDefault(),$=async()=>{if(!x){d("Log in om te genereren");return}if(!N){ue("Please upload an image first");return}re(!0),ue(null),ge(null),G(null);try{const _=new FormData;_.append("image",N),_.append("target_width",P.width),_.append("target_height",P.height),_.append("position",A),_.append("prompt",U||"seamless natural extension, high quality"),_.append("model",L.file),_.append("steps",ee),_.append("cfg",K),_.append("denoise",k),_.append("feathering",h);const Y=await ht(`${ve}/reframe`,_);if(!Y.ok)throw new Error(Y.data?.detail||"Reframe request failed");Y.data?.prompt_id?(G({promptId:Y.data.prompt_id,aspectRatio:P.label}),c&&c({prompt_id:Y.data.prompt_id})):Y.data?.url&&ge({url:Y.data.url})}catch(_){console.error("❌ Reframe error:",_),ue(_.message)}finally{re(!1)}},q=()=>{if(!xe?.url)return;const _=document.createElement("a");_.href=xe.url,_.download=`reframed_${P.id.replace(":","x")}_${Date.now()}.png`,_.click()},F=(()=>{if(!R.width||!R.height)return null;const _=P.width,Y=P.height,Q=R.width,u=R.height,he=_/Q,ze=Y/u,pe=Math.min(he,ze),Ne=Math.round(Q*pe),Re=Math.round(u*pe);let Ve=0,it=0;return A.includes("left")?Ve=0:A.includes("right")?Ve=_-Ne:Ve=(_-Ne)/2,A.includes("top")?it=0:A.includes("bottom")?it=Y-Re:it=(Y-Re)/2,{scaledW:Ne,scaledH:Re,offsetX:Ve,offsetY:it,targetW:_,targetH:Y}})();return t.jsxs("div",{className:"space-y-4",children:[t.jsxs("div",{onClick:()=>X.current?.click(),onDrop:J,onDragOver:m,className:"border-2 border-dashed border-gray-600 rounded-lg p-6 text-center cursor-pointer hover:border-purple-500 transition-colors",children:[t.jsx("input",{ref:X,type:"file",accept:"image/*",onChange:J,className:"hidden"}),b?t.jsxs("div",{className:"flex flex-col items-center gap-2",children:[t.jsx("img",{src:b,alt:"Preview",className:"max-h-32 rounded"}),t.jsxs("span",{className:"text-sm text-gray-400",children:["Original: ",R.width,"×",R.height]}),t.jsx("span",{className:"text-xs text-gray-500",children:"Click to change"})]}):t.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[t.jsx(pt,{className:"w-8 h-8"}),t.jsx("span",{children:"Drop image here or click to upload"})]})]}),t.jsxs("div",{children:[t.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Target Aspect Ratio"}),t.jsx("div",{className:"grid grid-cols-4 gap-2",children:ru.map(_=>t.jsx("button",{onClick:()=>T(_),className:`px-3 py-2 text-sm rounded transition-colors ${P.id===_.id?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:_.label},_.id))}),t.jsxs("span",{className:"text-xs text-gray-500 mt-1 block",children:["Output: ",P.width,"×",P.height]})]}),t.jsxs("div",{children:[t.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Image Position"}),t.jsx("div",{className:"grid grid-cols-3 gap-2 w-40 mx-auto",children:["top-left","top","top-right","left","center","right","bottom-left","bottom","bottom-right"].map(_=>t.jsx("button",{onClick:()=>C(_),className:`p-2 text-lg rounded transition-colors ${A===_?"bg-purple-600":"bg-gray-700 hover:bg-gray-600"}`,title:_,children:Hh.find(Y=>Y.id===_)?.icon||"○"},_))})]}),F&&t.jsxs("div",{className:"bg-gray-800 rounded-lg p-4",children:[t.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Layout Preview"}),t.jsxs("div",{className:"relative mx-auto border border-gray-600 bg-gray-900",style:{width:Math.min(300,F.targetW/3),height:Math.min(300,F.targetH/3),aspectRatio:`${F.targetW} / ${F.targetH}`},children:[t.jsx("div",{className:"absolute inset-0 bg-stripes opacity-30"}),t.jsx("div",{className:"absolute bg-purple-600/50 border-2 border-purple-400 flex items-center justify-center text-xs",style:{width:`${F.scaledW/F.targetW*100}%`,height:`${F.scaledH/F.targetH*100}%`,left:`${F.offsetX/F.targetW*100}%`,top:`${F.offsetY/F.targetH*100}%`},children:"Original"})]}),t.jsx("p",{className:"text-xs text-gray-500 text-center mt-2",children:"Purple = original image, striped = AI-generated fill"})]}),t.jsxs("div",{children:[t.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Fill Prompt (optional)"}),t.jsx("textarea",{value:U,onChange:_=>D(_.target.value),placeholder:"Describe what should appear in the extended areas...",className:"w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 resize-none",rows:2})]}),t.jsxs("div",{children:[t.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Model"}),t.jsx("div",{className:"flex gap-2",children:su.map(_=>t.jsx("button",{onClick:()=>V(_),className:`flex-1 px-3 py-2 text-sm rounded transition-colors ${L.id===_.id?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:_.label},_.id))})]}),t.jsxs("div",{className:"border border-gray-700 rounded-lg overflow-hidden",children:[t.jsxs("button",{onClick:()=>ie(!fe),className:"w-full px-4 py-2 bg-gray-800 flex items-center justify-between text-gray-300 hover:bg-gray-750",children:[t.jsx("span",{className:"text-sm font-medium",children:"Advanced Settings"}),t.jsx(Qt,{className:`w-4 h-4 transition-transform ${fe?"rotate-180":""}`})]}),fe&&t.jsxs("div",{className:"p-4 space-y-4 bg-gray-850",children:[t.jsxs("div",{children:[t.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Steps: ",ee]}),t.jsx("input",{type:"range",min:10,max:50,value:ee,onChange:_=>Z(Number(_.target.value)),className:"w-full accent-purple-500"})]}),t.jsxs("div",{children:[t.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["CFG Scale: ",K]}),t.jsx("input",{type:"range",min:1,max:15,step:.5,value:K,onChange:_=>j(Number(_.target.value)),className:"w-full accent-purple-500"})]}),t.jsxs("div",{children:[t.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Denoise: ",k.toFixed(2)]}),t.jsx("input",{type:"range",min:.5,max:1,step:.05,value:k,onChange:_=>B(Number(_.target.value)),className:"w-full accent-purple-500"}),t.jsx("span",{className:"text-xs text-gray-500",children:"Higher = more creative fill"})]}),t.jsxs("div",{children:[t.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Edge Feathering: ",h,"px"]}),t.jsx("input",{type:"range",min:0,max:64,step:8,value:h,onChange:_=>v(Number(_.target.value)),className:"w-full accent-purple-500"}),t.jsx("span",{className:"text-xs text-gray-500",children:"Blend between original and fill"})]})]})]}),t.jsx("button",{onClick:$,disabled:te||!N,className:"w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors",children:te?t.jsxs(t.Fragment,{children:[t.jsx(at,{className:"w-5 h-5 animate-spin"}),"Queueing..."]}):t.jsxs(t.Fragment,{children:[t.jsx(hm,{className:"w-5 h-5"}),"Reframe Image"]})}),W&&t.jsxs("div",{className:"p-3 bg-green-900/50 border border-green-700 rounded-lg text-green-200 text-sm",children:["✅ Reframe job queued! (",W.aspectRatio,") - Check queue panel for progress"]}),E&&t.jsx("div",{className:"p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm",children:E}),xe&&t.jsxs("div",{className:"space-y-3",children:[t.jsx("div",{className:"rounded-lg overflow-hidden border border-gray-700",children:t.jsx("img",{src:xe.url,alt:"Reframed",className:"w-full"})}),t.jsxs("div",{className:"flex gap-2",children:[t.jsxs("button",{onClick:q,className:"flex-1 py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center justify-center gap-2",children:[t.jsx(qt,{className:"w-4 h-4"}),"Download"]}),t.jsxs("button",{onClick:()=>{p(null),S(null),ge(null),fetch(xe.url).then(_=>_.blob()).then(_=>{const Y=new File([_],"reframed.png",{type:"image/png"});p(Y),S(xe.url);const Q=new Image;Q.onload=()=>I({width:Q.naturalWidth,height:Q.naturalHeight}),Q.src=xe.url})},className:"flex-1 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center justify-center gap-2",children:[t.jsx(Dm,{className:"w-4 h-4"}),"Use as Input"]})]})]}),t.jsxs("div",{className:"text-xs text-gray-500 space-y-1",children:[t.jsxs("p",{children:["💡 ",t.jsx("strong",{children:"Reframe"})," extends your image to a new aspect ratio using AI outpainting."]}),t.jsx("p",{children:"📐 The original image will be placed according to the position you select."}),t.jsx("p",{children:"🎨 Use the prompt to guide what should appear in the extended areas."})]})]})}const au=[{id:"inswapper",label:"InSwapper (Best Quality)",description:"High quality, slower"},{id:"simswap",label:"SimSwap (Fast)",description:"Faster, good quality"}],Qh=[{id:"none",label:"None"},{id:"gfpgan",label:"GFPGAN (Faces)"},{id:"codeformer",label:"CodeFormer (Natural)"},{id:"both",label:"Both (Best)"}];function Yh({onJobSubmitted:c}){const{user:x,requestLogin:d}=Ke(),[N,p]=l.useState(null),[b,S]=l.useState(null),[R,I]=l.useState(null),[P,T]=l.useState(null),[A,C]=l.useState(au[0]),[L,V]=l.useState("gfpgan"),[U,D]=l.useState(1),[ee,Z]=l.useState(.8),[K,j]=l.useState(0),[k,B]=l.useState(!1),[h,v]=l.useState(!1),[te,re]=l.useState(null),[xe,ge]=l.useState(null),[E,ue]=l.useState(null),[fe,ie]=l.useState(!1),[W,G]=l.useState(null),X=l.useRef(null),J=l.useRef(null),m=l.useCallback(Q=>{Q.preventDefault();const u=Q.dataTransfer?.files?.[0]||Q.target?.files?.[0];if(u&&(u.type.startsWith("image/")||u.type.startsWith("video/"))){p(u),re(null),ge(null),ue(null),G(null);const he=URL.createObjectURL(u);S(he)}},[]),$=l.useCallback(Q=>{Q.preventDefault();const u=Q.dataTransfer?.files?.[0]||Q.target?.files?.[0];if(u&&u.type.startsWith("image/")){I(u),re(null),ge(null),G(null);const he=URL.createObjectURL(u);T(he)}},[]),q=Q=>Q.preventDefault(),le=async()=>{if(N){v(!0),ge(null);try{const Q=new FormData;Q.append("image",N);const u=await ht(`${ve}/detect-faces`,Q);if(u.ok&&u.data?.faces)ue(u.data.faces);else throw new Error(u.data?.detail||"Face detection failed")}catch(Q){console.error("❌ Face detection error:",Q),ge(Q.message)}finally{v(!1)}}},F=async()=>{if(!x){d("Log in om te genereren");return}if(!N||!R){ge("Please upload both target and source face images");return}v(!0),ge(null),re(null),G(null);try{const Q=new FormData;Q.append("target",N),Q.append("source",R),Q.append("model",A.id),Q.append("enhance",L),Q.append("strength",U),Q.append("blend",ee),Q.append("face_index",k?-1:K);const u=await ht(`${ve}/face-swap`,Q);if(!u.ok)throw new Error(u.data?.detail||"Face swap request failed");u.data?.prompt_id?(G({promptId:u.data.prompt_id,model:A.label}),c&&c({prompt_id:u.data.prompt_id})):u.data?.url&&re({url:u.data.url})}catch(Q){console.error("❌ FaceSwap error:",Q),ge(Q.message)}finally{v(!1)}},_=()=>{if(!te?.url)return;const Q=N?.type.startsWith("video/")?"mp4":"png",u=document.createElement("a");u.href=te.url,u.download=`face_swap_${Date.now()}.${Q}`,u.click()},Y=()=>{const Q=N,u=b;p(R),S(P),I(Q),T(u),re(null),ue(null),G(null)};return t.jsxs("div",{className:"space-y-4",children:[t.jsxs("div",{className:"grid grid-cols-2 gap-4",children:[t.jsxs("div",{children:[t.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Target (face to replace)"}),t.jsxs("div",{onClick:()=>X.current?.click(),onDrop:m,onDragOver:q,className:"border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-purple-500 transition-colors aspect-square flex items-center justify-center",children:[t.jsx("input",{ref:X,type:"file",accept:"image/*,video/*",onChange:m,className:"hidden"}),b?t.jsxs("div",{className:"relative w-full h-full",children:[N?.type.startsWith("video/")?t.jsx("video",{src:b,className:"w-full h-full object-cover rounded",muted:!0}):t.jsx("img",{src:b,alt:"Target",className:"w-full h-full object-cover rounded"}),E&&t.jsxs("div",{className:"absolute bottom-1 right-1 bg-black/70 px-2 py-1 rounded text-xs",children:[E.length," face",E.length!==1?"s":""," detected"]})]}):t.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[t.jsx(pt,{className:"w-6 h-6"}),t.jsx("span",{className:"text-xs",children:"Target image/video"})]})]})]}),t.jsxs("div",{children:[t.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Source (face to use)"}),t.jsxs("div",{onClick:()=>J.current?.click(),onDrop:$,onDragOver:q,className:"border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-blue-500 transition-colors aspect-square flex items-center justify-center",children:[t.jsx("input",{ref:J,type:"file",accept:"image/*",onChange:$,className:"hidden"}),P?t.jsx("img",{src:P,alt:"Source",className:"w-full h-full object-cover rounded"}):t.jsxs("div",{className:"flex flex-col items-center gap-2 text-gray-400",children:[t.jsx(rx,{className:"w-6 h-6"}),t.jsx("span",{className:"text-xs",children:"Source face"})]})]})]})]}),(N||R)&&t.jsxs("button",{onClick:Y,className:"w-full py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center justify-center gap-2 text-sm",children:[t.jsx(_s,{className:"w-4 h-4"}),"Swap Target ↔ Source"]}),N&&!N.type.startsWith("video/")&&t.jsxs("button",{onClick:le,disabled:h,className:"w-full py-2 bg-gray-700 hover:bg-gray-600 rounded-lg flex items-center justify-center gap-2 text-sm",children:[t.jsx(Gl,{className:"w-4 h-4"}),"Detect Faces"]}),E&&E.length>1&&t.jsxs("div",{className:"bg-gray-800 rounded-lg p-3 space-y-2",children:[t.jsx("label",{className:"block text-sm font-medium text-gray-300",children:"Select Face to Replace"}),t.jsx("div",{className:"flex items-center gap-4",children:t.jsxs("label",{className:"flex items-center gap-2",children:[t.jsx("input",{type:"checkbox",checked:k,onChange:Q=>B(Q.target.checked),className:"rounded bg-gray-700 border-gray-600"}),t.jsx("span",{className:"text-sm text-gray-300",children:"Swap all faces"})]})}),!k&&t.jsx("div",{className:"flex gap-2 flex-wrap",children:E.map((Q,u)=>t.jsxs("button",{onClick:()=>j(u),className:`px-3 py-1 text-sm rounded ${K===u?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:["Face ",u+1]},u))})]}),t.jsxs("div",{children:[t.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Model"}),t.jsx("div",{className:"space-y-2",children:au.map(Q=>t.jsxs("button",{onClick:()=>C(Q),className:`w-full px-3 py-2 text-left rounded transition-colors ${A.id===Q.id?"bg-purple-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:[t.jsx("div",{className:"font-medium text-sm",children:Q.label}),t.jsx("div",{className:"text-xs opacity-70",children:Q.description})]},Q.id))})]}),t.jsxs("div",{children:[t.jsx("label",{className:"block text-sm font-medium text-gray-300 mb-2",children:"Face Enhancement"}),t.jsx("div",{className:"grid grid-cols-2 gap-2",children:Qh.map(Q=>t.jsx("button",{onClick:()=>V(Q.id),className:`px-3 py-2 text-sm rounded transition-colors ${L===Q.id?"bg-blue-600 text-white":"bg-gray-700 text-gray-300 hover:bg-gray-600"}`,children:Q.label},Q.id))})]}),t.jsxs("div",{className:"border border-gray-700 rounded-lg overflow-hidden",children:[t.jsxs("button",{onClick:()=>ie(!fe),className:"w-full px-4 py-2 bg-gray-800 flex items-center justify-between text-gray-300 hover:bg-gray-750",children:[t.jsx("span",{className:"text-sm font-medium",children:"Advanced Settings"}),t.jsx(Qt,{className:`w-4 h-4 transition-transform ${fe?"rotate-180":""}`})]}),fe&&t.jsxs("div",{className:"p-4 space-y-4 bg-gray-850",children:[t.jsxs("div",{children:[t.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Swap Strength: ",U.toFixed(2)]}),t.jsx("input",{type:"range",min:.1,max:1,step:.05,value:U,onChange:Q=>D(Number(Q.target.value)),className:"w-full accent-purple-500"}),t.jsx("span",{className:"text-xs text-gray-500",children:"Lower = more original features preserved"})]}),t.jsxs("div",{children:[t.jsxs("label",{className:"block text-sm text-gray-400 mb-1",children:["Edge Blend: ",ee.toFixed(2)]}),t.jsx("input",{type:"range",min:0,max:1,step:.05,value:ee,onChange:Q=>Z(Number(Q.target.value)),className:"w-full accent-purple-500"}),t.jsx("span",{className:"text-xs text-gray-500",children:"Blend face edges with background"})]})]})]}),t.jsxs("div",{className:"flex items-start gap-2 p-3 bg-yellow-900/30 border border-yellow-700/50 rounded-lg",children:[t.jsx(Yl,{className:"w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5"}),t.jsxs("div",{className:"text-sm text-yellow-200",children:[t.jsx("strong",{children:"Ethical Use:"})," Only use face swap with consent of all parties involved. Creating non-consensual deepfakes is illegal in many jurisdictions."]})]}),t.jsx("button",{onClick:F,disabled:h||!N||!R,className:"w-full py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors",children:h?t.jsxs(t.Fragment,{children:[t.jsx(at,{className:"w-5 h-5 animate-spin"}),"Swapping... ",progress>0&&`${Math.round(progress)}%`]}):t.jsxs(t.Fragment,{children:[t.jsx(Gl,{className:"w-5 h-5"}),"Swap Face"]})}),xe&&t.jsx("div",{className:"p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm",children:xe}),te&&t.jsxs("div",{className:"space-y-3",children:[t.jsx("div",{className:"rounded-lg overflow-hidden border border-gray-700",children:N?.type.startsWith("video/")?t.jsx("video",{src:te.url,controls:!0,className:"w-full"}):t.jsx("img",{src:te.url,alt:"Result",className:"w-full"})}),t.jsxs("button",{onClick:_,className:"w-full py-2 bg-green-600 hover:bg-green-700 rounded-lg flex items-center justify-center gap-2",children:[t.jsx(qt,{className:"w-4 h-4"}),"Download Result"]})]}),t.jsxs("div",{className:"text-xs text-gray-500 space-y-1",children:[t.jsxs("p",{children:["👤 ",t.jsx("strong",{children:"Face Swap"})," replaces faces in images or videos using AI."]}),t.jsx("p",{children:"📸 For best results, use clear frontal face photos with good lighting."}),t.jsx("p",{children:"🎬 Video processing may take longer depending on length and resolution."})]})]})}function Xh({title:c}){return t.jsxs("div",{className:"tool-coming-soon",children:[t.jsx("div",{className:"tool-title",children:c}),t.jsx("div",{className:"muted",children:"Missing backend endpoint (planned for v2)."})]})}function fr(c){if(!c)return"image";const x=c.toLowerCase().split(".").pop();return!x||x===c.toLowerCase()?"image":["mp4","webm","mov","avi","mkv","flv"].includes(x)?"video":["jpg","jpeg","png","gif","webp","bmp","svg"].includes(x)?"image":["mp3","wav","ogg","flac","aac","m4a"].includes(x)?"audio":"image"}function Kh({mediaItem:c,onClose:x,onPublished:d}){const N=c.metadata?.positive_prompt?c.metadata.positive_prompt.slice(0,100):"Untitled Creation",[p,b]=l.useState(N),[S,R]=l.useState(""),[I,P]=l.useState(""),[T,A]=l.useState(!1),[C,L]=l.useState(!1),[V,U]=l.useState(""),D=()=>{if(c.source==="storage"){const j=fr(c.filename);return`${ve}/user/media/${j}/${c.filename}`}else return`${ve}/comfyui/output/${c.filename}`},ee=async()=>{if(!p.trim()){U("Title is required");return}if(p.length>100){U("Title must be 100 characters or less");return}if(S.length>500){U("Description must be 500 characters or less");return}L(!0),U("");try{const j=I.split(",").map(re=>re.trim()).filter(re=>re.length>0).slice(0,10),k=fr(c.filename),h={storage_path:`${k}/${c.filename}`,title:p.trim(),description:S.trim()||null,tags:j,is_nsfw:T,media_type:k,thumbnail_url:null,metadata:c.metadata||{}},v=await rn("/api/gallery/publish",{method:"POST",body:JSON.stringify(h)});if(!v.ok){let re="Failed to publish";try{const xe=await v.json();xe?.detail?re=xe.detail:xe?.message&&(re=xe.message)}catch{try{const ge=await v.text();ge&&ge.trim()&&(re=ge)}catch{}}throw new Error(re)}const te=await v.json();console.log("✅ Published successfully:",te),d&&d(te),x()}catch(j){console.error("❌ Publish error:",j),U(j.message||"Failed to publish media")}finally{L(!1)}},Z=D(),K=fr(c.filename);return t.jsx("div",{className:"modal-overlay",onClick:j=>{j.target===j.currentTarget&&x()},children:t.jsxs("div",{className:"modal-content",onClick:j=>j.stopPropagation(),style:{maxWidth:"600px",maxHeight:"90vh",overflowY:"auto"},children:[t.jsxs("div",{className:"modal-header",style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"16px",paddingBottom:"12px",borderBottom:"1px solid #333"},children:[t.jsxs("h3",{style:{margin:0,display:"flex",alignItems:"center",gap:"8px"},children:[t.jsx(pt,{size:20}),"Publish to Gallery"]}),t.jsx("button",{onClick:x,style:{background:"none",border:"none",color:"#ccc",cursor:"pointer",padding:"4px",display:"flex"},children:t.jsx(lt,{size:20})})]}),t.jsx("div",{style:{marginBottom:"16px",borderRadius:"8px",overflow:"hidden",background:"#000"},children:K==="video"?t.jsx("video",{src:Z,controls:!0,style:{width:"100%",maxHeight:"300px",objectFit:"contain"}}):t.jsx("img",{src:Z,alt:"Preview",style:{width:"100%",maxHeight:"300px",objectFit:"contain"}})}),V&&t.jsxs("div",{style:{marginBottom:"16px",padding:"12px",background:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.3)",borderRadius:"6px",color:"#ef4444",display:"flex",alignItems:"flex-start",gap:"8px"},children:[t.jsx(Yl,{size:18,style:{marginTop:"2px",flexShrink:0}}),t.jsx("span",{children:V})]}),t.jsxs("div",{style:{display:"flex",flexDirection:"column",gap:"16px"},children:[t.jsxs("div",{children:[t.jsxs("label",{style:{display:"block",marginBottom:"6px",fontSize:"13px",color:"#ccc"},children:["Title ",t.jsx("span",{style:{color:"#ef4444"},children:"*"})]}),t.jsx("input",{type:"text",value:p,onChange:j=>b(j.target.value),placeholder:"Give your creation a catchy title...",maxLength:100,style:{width:"100%",padding:"8px 12px",background:"#1a1a1a",border:"1px solid #333",borderRadius:"6px",color:"#fff",fontSize:"14px"}}),t.jsxs("div",{style:{fontSize:"11px",color:"#666",marginTop:"4px",textAlign:"right"},children:[p.length,"/100"]})]}),t.jsxs("div",{children:[t.jsx("label",{style:{display:"block",marginBottom:"6px",fontSize:"13px",color:"#ccc"},children:"Description (optional)"}),t.jsx("textarea",{value:S,onChange:j=>R(j.target.value),placeholder:"Add a description to help people understand your creation...",maxLength:500,rows:4,style:{width:"100%",padding:"8px 12px",background:"#1a1a1a",border:"1px solid #333",borderRadius:"6px",color:"#fff",fontSize:"14px",resize:"vertical"}}),t.jsxs("div",{style:{fontSize:"11px",color:"#666",marginTop:"4px",textAlign:"right"},children:[S.length,"/500"]})]}),t.jsxs("div",{children:[t.jsxs("label",{style:{marginBottom:"6px",fontSize:"13px",color:"#ccc",display:"flex",alignItems:"center",gap:"6px"},children:[t.jsx(ox,{size:14}),"Tags (comma-separated, max 10)"]}),t.jsx("input",{type:"text",value:I,onChange:j=>P(j.target.value),placeholder:"e.g., anime, portrait, fantasy",style:{width:"100%",padding:"8px 12px",background:"#1a1a1a",border:"1px solid #333",borderRadius:"6px",color:"#fff",fontSize:"14px"}})]}),t.jsx("div",{style:{padding:"12px",background:"#1a1a1a",border:"1px solid #333",borderRadius:"6px"},children:t.jsxs("label",{style:{display:"flex",alignItems:"center",gap:"12px",cursor:"pointer",fontSize:"14px"},children:[t.jsx("input",{type:"checkbox",checked:T,onChange:j=>A(j.target.checked),style:{width:"18px",height:"18px",cursor:"pointer"}}),t.jsxs("span",{children:[t.jsx("span",{style:{fontWeight:"500"},children:"Mark as NSFW"}),t.jsx("span",{style:{display:"block",fontSize:"12px",color:"#888",marginTop:"2px"},children:"Content will only be visible to logged-in users"})]})]})})]}),t.jsxs("div",{style:{display:"flex",gap:"12px",marginTop:"24px",paddingTop:"16px",borderTop:"1px solid #333"},children:[t.jsx("button",{onClick:x,disabled:C,style:{flex:1,padding:"10px 20px",background:"#2a2a2a",border:"1px solid #444",borderRadius:"6px",color:"#ccc",fontSize:"14px",fontWeight:"500",cursor:C?"not-allowed":"pointer",opacity:C?.5:1},children:"Cancel"}),t.jsx("button",{onClick:ee,disabled:C||!p.trim(),style:{flex:1,padding:"10px 20px",background:C||!p.trim()?"#444":"#3b82f6",border:"none",borderRadius:"6px",color:"#fff",fontSize:"14px",fontWeight:"500",cursor:C||!p.trim()?"not-allowed":"pointer",opacity:C?.7:1},children:C?"Publishing...":"Publish"})]})]})})}const ou=c=>{if(!c||isNaN(c))return null;const x=Math.floor(c/60),d=Math.floor(c%60);return`${x}:${d.toString().padStart(2,"0")}`},Su="oelala_media_favorites",Nu="oelala_media_profile",Aa={"1280x1024":{cols:4,label:"1280×1024"},"1080p":{cols:5,label:"1080p"},"1440p":{cols:6,label:"1440p"},"4k":{cols:8,label:"4K"}},lu=()=>{const c=window.innerWidth;return c<=1280?"1280x1024":c<=1920?"1080p":c<=2560?"1440p":"4k"},Jh=()=>{try{return localStorage.getItem(Nu)||"auto"}catch{return"auto"}},Al=c=>{try{localStorage.setItem(Nu,c)}catch(x){console.error("Failed to save profile:",x)}},Zh=()=>{try{const c=localStorage.getItem(Su);return c?new Set(JSON.parse(c)):new Set}catch{return new Set}},eg=c=>{try{localStorage.setItem(Su,JSON.stringify([...c]))}catch(x){console.error("Failed to save favorites:",x)}};function $r({filter:c="all",selectionMode:x=!1,onSelectItem:d=null}){const[N,p]=l.useState([]),[b,S]=l.useState(!1),[R,I]=l.useState(""),[P,T]=l.useState({videos:0,images:0,audio:0}),[A,C]=l.useState(null),[L,V]=l.useState(new Set),[U,D]=l.useState(null),[ee,Z]=l.useState(!1),[K,j]=l.useState(!1),[k,B]=l.useState(null),[h,v]=l.useState(Zh),[te,re]=l.useState("date"),[xe,ge]=l.useState("desc"),[E,ue]=l.useState("all"),[fe,ie]=l.useState(""),[W,G]=l.useState(!0),[X,J]=l.useState(Jh),[m,$]=l.useState(null),[q,le]=l.useState(new Set),F=X==="auto"?lu():X,Y=(Aa[F]||Aa["1080p"]).cols,[Q,u]=l.useState(!1),[he,ze]=l.useState(100),[pe,Ne]=l.useState(320),[Re,Ve]=l.useState({}),it=l.useRef(null),{user:Je}=Ke();l.useEffect(()=>{if(!Je)return;(async()=>{try{const ce=await rn(`/api/gallery/users/${Je.id}?per_page=100`);if(ce.ok){const ae=await ce.json(),ke=new Set(ae.items.map(Ae=>Ae.storage_path));le(ke)}}catch(ce){console.error("Failed to fetch published items:",ce)}})()},[Je]),l.useEffect(()=>{const w=()=>{if(it.current){const ke=(it.current.clientWidth-32-12*(Y-1))/Y,Ae=Math.round(ke*(16/9));Ne(Ae)}};return w(),window.addEventListener("resize",w),()=>window.removeEventListener("resize",w)},[Y]),l.useEffect(()=>{ze(100)},[E,te,xe,N]);const Kn=w=>{const{scrollTop:ce,clientHeight:ae,scrollHeight:ke}=w.target;ke-ce-ae<1e3&&ze(Ae=>Math.min(Ae+50,We.length))},_t=(w,ce)=>{ce?.stopPropagation(),v(ae=>{const ke=new Set(ae);return ke.has(w)?ke.delete(w):ke.add(w),eg(ke),ke})},We=l.useMemo(()=>{let w=[...N];if(E==="favorites"?w=w.filter(ce=>h.has(ce.filename)):E==="non-favorites"&&(w=w.filter(ce=>!h.has(ce.filename))),fe.trim()){const ce=fe.toLowerCase().trim();w=w.filter(ae=>!!(ae.filename.toLowerCase().includes(ce)||ae.metadata?.positive_prompt?.toLowerCase().includes(ce)||ae.metadata?.prompt?.toLowerCase().includes(ce)||ae.metadata?.negative_prompt?.toLowerCase().includes(ce)))}return w.sort((ce,ae)=>{let ke=0;switch(te){case"name":ke=ce.filename.localeCompare(ae.filename);break;case"size":ke=(ce.size||0)-(ae.size||0);break;case"favorites":const Ae=h.has(ce.filename)?1:0,Oe=h.has(ae.filename)?1:0;ke=Ae-Oe;break;case"non-favorites":const ct=h.has(ce.filename)?0:1,St=h.has(ae.filename)?0:1;ke=ct-St;break;default:ke=(ce.mtime||0)-(ae.mtime||0);break}return xe==="desc"?-ke:ke}),w},[N,te,xe,E,h,fe]),kt=l.useCallback(async()=>{S(!0),I("");try{const w=c==="prompts"?"all":c,ae=Je&&["mark.op.mobiel@gmail.com"].includes(Je.email);console.log("🎬 MyMedia: Fetching media, user:",Je?.id,Je?.email,"isAdmin:",ae);let ke={media:[],stats:{videos:0,images:0,audio:0}},Ae={media:[],stats:{videos:0,images:0,audio:0}};Je?(Ae=await Ox(w==="video"?"video":w==="image"?"image":"all").then(Ge=>(console.log("🎬 MyMedia: User storage response:",Ge),Ge)).catch(Ge=>(console.error("🎬 MyMedia: User storage error:",Ge),{media:[],stats:{videos:0,images:0,audio:0}})),ae&&(ke=await fetch(`${ve}/list-comfyui-media?type=${w}&grouped=true&include_metadata=true&hide_start_images=${W}`).then(Ge=>Ge.ok?Ge.json():{media:[],stats:{videos:0,images:0,audio:0}}).catch(()=>({media:[],stats:{videos:0,images:0,audio:0}})))):ke={media:[],stats:{videos:0,images:0,audio:0}};const Oe=(Ae.media||[]).map(Ge=>({...Ge,source:"storage"})),ct=(ke.media||[]).map(Ge=>({...Ge,source:"comfyui"}));let St=[...Oe,...ct];c==="prompts"&&(St=St.filter(Ge=>Ge.metadata?.positive_prompt||Ge.metadata?.prompt));const Qe={videos:(ke.stats?.videos||0)+(Ae.stats?.videos||0),images:(ke.stats?.images||0)+(Ae.stats?.images||0),audio:(ke.stats?.audio||0)+(Ae.stats?.audio||0)};p(St),T(Qe),V(new Set)}catch(w){I(w.message)}finally{S(!1)}},[c,W,Je?.id]);l.useEffect(()=>{let w=!0;return w&&kt(),()=>{w=!1}},[c,W,Je?.id]),l.useEffect(()=>{const w=ce=>{if(ce.key==="?"||ce.key==="/"&&ce.shiftKey){ce.preventDefault(),u(ae=>!ae);return}if(ce.key==="+"||ce.key==="="){ce.preventDefault();const ae=["auto","1280x1024","1080p","1440p","4k"];J(ke=>{const Ae=ae.indexOf(ke),Oe=ae[(Ae+1)%ae.length];return Al(Oe),Oe});return}if(ce.key==="-"||ce.key==="_"){ce.preventDefault();const ae=["auto","1280x1024","1080p","1440p","4k"];J(ke=>{const Ae=ae.indexOf(ke),Oe=ae[(Ae-1+ae.length)%ae.length];return Al(Oe),Oe});return}if(A!==null&&(ce.key==="Escape"&&(C(null),u(!1)),ce.key==="ArrowLeft"&&C(ae=>ae>0?ae-1:We.length-1),ce.key==="ArrowRight"&&C(ae=>ae<We.length-1?ae+1:0),ce.key==="f"||ce.key==="F"||ce.key==="h"||ce.key==="H")){const ae=We[A];ae&&_t(ae.filename)}};return window.addEventListener("keydown",w),()=>{window.removeEventListener("keydown",w)}},[A,We,h]);const _n=(w,ce)=>{if(ce.target.closest(".select-checkbox")){ce.stopPropagation(),pn(w,ce);return}if(x&&d){const ae=We[w];d(ae);return}C(w)},pn=(w,ce)=>{ce?.stopPropagation(),V(ae=>{const ke=new Set(ae);if(ce?.shiftKey&&U!==null){const Ae=Math.min(U,w),Oe=Math.max(U,w);for(let ct=Ae;ct<=Oe;ct++)ke.add(ct)}else ce?.ctrlKey||ce?.metaKey,ke.has(w)?ke.delete(w):ke.add(w);return ke}),D(w)},hr=()=>{V(new Set(N.map((w,ce)=>ce)))},zn=()=>{V(new Set)},gr=async()=>{if(L.size===0)return;const w=Array.from(L).map(Oe=>We[Oe]?.filename).filter(Boolean);if(w.length===0){I("No valid items selected for deletion");return}const ce=w.filter(Oe=>h.has(Oe)),ae=ce.length;let ke=`Delete ${w.length} item${w.length>1?"s":""} and their associated files (source images, metadata)?`;if(ae>0&&(ke=`⚠️ WARNING: ${ae} favorite${ae>1?"s":""} selected!

${ke}

Favorites to delete:
• ${ce.slice(0,5).join(`
• `)}${ae>5?`
• ... and ${ae-5} more`:""}`),!!window.confirm(ke)){Z(!0);try{const Oe=Array.from(L).map(Qe=>We[Qe]).filter(Boolean),ct=Oe.filter(Qe=>Qe.source==="comfyui"||!Qe.source),St=Oe.filter(Qe=>Qe.source==="storage");if(ct.length>0){const Qe=ct.map(Lt=>Lt.filename);(await rn(`${ve}/delete-comfyui-media`,{method:"DELETE",headers:{"Content-Type":"application/json"},body:JSON.stringify({filenames:Qe})})).ok||console.error("Failed to delete some ComfyUI items")}for(const Qe of St)try{const Lt=(Qe.url||"").split("/")[3]||"images";await Ax(Lt,Qe.name||Qe.filename)}catch(Ge){console.error(`Failed to delete storage item ${Qe.filename}:`,Ge)}await kt()}catch(Oe){I(`Delete failed: ${Oe.message}`)}finally{Z(!1)}}},Mt=(w,ce)=>{ce?.stopPropagation();const ae=document.createElement("a");ae.href=`${ve}${w.url}`,ae.download=w.filename,ae.click()},fn=async()=>{if(L.size===0)return;const w=We.filter(ce=>L.has(ce.filename));for(let ce=0;ce<w.length;ce++){const ae=w[ce],ke=document.createElement("a");ke.href=`${ve}${ae.url}`,ke.download=ae.filename,ke.click(),ce<w.length-1&&await new Promise(Ae=>setTimeout(Ae,300))}},mn=async(w,ce)=>{ce?.stopPropagation();try{const ae=await fetch(`${ve}/comfyui-metadata/${w.filename}`);if(!ae.ok)throw new Error("No metadata available");const ke=await ae.json(),Ae=new Blob([JSON.stringify(ke.metadata,null,2)],{type:"application/json"}),Oe=URL.createObjectURL(Ae),ct=document.createElement("a");ct.href=Oe,ct.download=`${w.base_name||w.filename.replace(/\.[^/.]+$/,"")}_metadata.json`,ct.click(),URL.revokeObjectURL(Oe)}catch(ae){console.error("Failed to download metadata:",ae)}},$t=w=>w<1024?`${w} B`:w<1024*1024?`${(w/1024).toFixed(1)} KB`:`${(w/1024/1024).toFixed(1)} MB`,Fe=A!==null?We[A]:null,Wr=N.filter(w=>h.has(w.filename)).length;return t.jsxs("div",{style:{display:"flex",flexDirection:"column",height:"100%",backgroundColor:"var(--bg-primary)"},children:[t.jsx("style",{children:`
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

        /* ========== PUBLISH BUTTON ========== */
        .publish-btn {
          position: absolute;
          top: 8px;
          left: 70px;
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
        .thumb-card:hover .publish-btn {
          opacity: 1;
        }
        .publish-btn.is-published {
          opacity: 1;
          background: #10b981;
          border-color: #10b981;
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
      `}),t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"12px 16px",borderBottom:"1px solid var(--border-color)",backgroundColor:"var(--bg-secondary)",flexWrap:"wrap",gap:"10px"},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"16px"},children:[t.jsx("span",{style:{fontWeight:600,color:"var(--text-primary)"},children:c==="all"?"All Media":c==="video"?"Videos":c==="image"?"Images":c==="audio"?"Audio":"Prompts"}),t.jsxs("span",{style:{color:"var(--text-muted)",fontSize:"0.85rem"},children:[c==="prompts"?t.jsxs(t.Fragment,{children:["💬 ",We.length," items with prompts"]}):t.jsxs(t.Fragment,{children:["🎬 ",P.videos," • 🖼️ ",P.images," • 🎵 ",P.audio," • ❤️ ",Wr]}),E!=="all"&&` • 📋 ${We.length} shown`]})]}),t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px"},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"4px",position:"relative"},children:[t.jsx(qm,{size:14,style:{color:"var(--text-muted)",position:"absolute",left:"8px"}}),t.jsx("input",{type:"text",placeholder:"Search filename or prompt...",value:fe,onChange:w=>ie(w.target.value),style:{background:"rgba(255,255,255,0.08)",border:"1px solid var(--border-color)",borderRadius:"6px",padding:"6px 8px 6px 28px",color:"var(--text-primary)",fontSize:"0.85rem",width:"200px",outline:"none"}}),fe&&t.jsx("button",{onClick:()=>ie(""),style:{position:"absolute",right:"6px",background:"none",border:"none",color:"var(--text-muted)",cursor:"pointer",padding:"2px"},children:t.jsx(lt,{size:12})})]}),t.jsx("div",{style:{width:"1px",height:"20px",background:"var(--border-color)",margin:"0 4px"}}),t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"4px"},children:[t.jsx(pu,{size:14,style:{color:"var(--text-muted)"}}),t.jsxs("select",{className:"sort-select",value:E,onChange:w=>{ue(w.target.value),V(new Set)},children:[t.jsx("option",{value:"all",children:"All"}),t.jsx("option",{value:"favorites",children:"❤️ Favorites"}),t.jsx("option",{value:"non-favorites",children:"🤍 Non-favorites"})]}),(c==="all"||c==="image")&&t.jsxs("button",{className:"sort-btn",onClick:()=>G(w=>!w),title:W?"Click to show video source images":"Hiding video source images",style:{background:W?void 0:"var(--accent-color, #a855f7)",color:W?void 0:"#fff",fontSize:"0.75rem",padding:"4px 8px"},children:["📸",W?"":"✓"]})]}),t.jsx("div",{style:{width:"1px",height:"20px",background:"var(--border-color)",margin:"0 4px"}}),t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"4px"},children:[t.jsx(Df,{size:14,style:{color:"var(--text-muted)"}}),t.jsxs("select",{className:"sort-select",value:te,onChange:w=>re(w.target.value),children:[t.jsx("option",{value:"date",children:"Date"}),t.jsx("option",{value:"name",children:"Name"}),t.jsx("option",{value:"size",children:"Size"}),t.jsx("option",{value:"favorites",children:"Favorites ❤️"}),t.jsx("option",{value:"non-favorites",children:"Non-favorites 🤍"})]}),t.jsx("button",{className:"sort-btn",onClick:()=>ge(w=>w==="asc"?"desc":"asc"),title:xe==="asc"?"Ascending":"Descending",children:xe==="asc"?"↑":"↓"})]}),t.jsx("div",{style:{width:"1px",height:"20px",background:"var(--border-color)",margin:"0 4px"}}),t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"2px"},children:[t.jsx("span",{style:{color:"var(--text-muted)",fontSize:"0.75rem",marginRight:"4px"},children:"Profile:"}),["auto","1280x1024","1080p","1440p","4k"].map(w=>t.jsx("button",{className:"sort-btn",onClick:()=>{J(w),Al(w)},title:w==="auto"?`Auto-detect (currently ${lu()})`:Aa[w]?.label||w,style:{background:X===w?"var(--accent-color, #a855f7)":void 0,color:X===w?"#fff":void 0,fontSize:"0.7rem",padding:"4px 6px"},children:w==="auto"?"⚡Auto":Aa[w]?.label||w},w)),t.jsxs("span",{style:{color:"var(--text-muted)",fontSize:"0.7rem",marginLeft:"8px"},children:[Y," cols"]})]}),t.jsx("div",{style:{width:"1px",height:"20px",background:"var(--border-color)",margin:"0 4px"}}),L.size>0&&t.jsxs(t.Fragment,{children:[t.jsxs("span",{style:{color:"var(--text-muted)",fontSize:"0.85rem"},children:[L.size," selected"]}),t.jsx("button",{className:"header-btn",onClick:zn,children:"Clear"}),t.jsx("button",{className:"header-btn",onClick:hr,children:"Select All"}),t.jsxs("button",{className:"header-btn",onClick:fn,title:"Download selected items",children:[t.jsx(qt,{size:16}),"Download"]}),t.jsxs("button",{className:"delete-btn",onClick:gr,disabled:ee,children:[t.jsx(Ba,{size:16}),ee?"Deleting...":"Delete"]})]}),t.jsx("button",{onClick:kt,disabled:b,style:{padding:"8px",borderRadius:"6px",border:"none",background:"transparent",color:"var(--text-muted)",cursor:"pointer",display:"flex",alignItems:"center"},title:"Refresh",children:t.jsx(_s,{size:18,style:{animation:b?"spin 1s linear infinite":"none"}})}),t.jsx("button",{onClick:()=>u(!0),style:{padding:"6px",border:"none",background:"transparent",color:"var(--text-muted)",cursor:"pointer",display:"flex",alignItems:"center"},title:"Keyboard shortcuts (?)",children:t.jsx(Kf,{size:18})})]})]}),Q&&t.jsx("div",{style:{position:"fixed",top:0,left:0,right:0,bottom:0,backgroundColor:"rgba(0,0,0,0.8)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:2e3},onClick:()=>u(!1),children:t.jsxs("div",{style:{backgroundColor:"var(--bg-primary, #1a1a1a)",borderRadius:"12px",padding:"24px",maxWidth:"500px",width:"90%",boxShadow:"0 20px 60px rgba(0,0,0,0.5)"},onClick:w=>w.stopPropagation(),children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"20px"},children:[t.jsx("h3",{style:{margin:0,color:"var(--text-primary, #fff)",fontSize:"1.2rem"},children:"⌨️ Keyboard Shortcuts"}),t.jsx("button",{onClick:()=>u(!1),style:{background:"transparent",border:"none",color:"var(--text-muted)",cursor:"pointer",padding:"4px"},children:t.jsx(lt,{size:20})})]}),t.jsxs("div",{style:{color:"var(--text-secondary, #ccc)",fontSize:"0.9rem"},children:[t.jsxs("div",{style:{marginBottom:"16px"},children:[t.jsx("div",{style:{color:"var(--accent-color, #a855f7)",fontWeight:600,marginBottom:"8px"},children:"Grid View"}),t.jsxs("div",{style:{display:"grid",gridTemplateColumns:"120px 1fr",gap:"6px 16px"},children:[t.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"+"}),t.jsx("span",{children:"More columns (smaller thumbnails)"}),t.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"-"}),t.jsx("span",{children:"Fewer columns (larger thumbnails)"}),t.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"?"}),t.jsx("span",{children:"Show this help"})]})]}),t.jsxs("div",{style:{marginBottom:"16px"},children:[t.jsx("div",{style:{color:"var(--accent-color, #a855f7)",fontWeight:600,marginBottom:"8px"},children:"Lightbox (Image View)"}),t.jsxs("div",{style:{display:"grid",gridTemplateColumns:"120px 1fr",gap:"6px 16px"},children:[t.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"←"}),t.jsx("span",{children:"Previous image"}),t.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"→"}),t.jsx("span",{children:"Next image"}),t.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"F / H"}),t.jsx("span",{children:"Toggle favorite ❤️"}),t.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"Esc"}),t.jsx("span",{children:"Close lightbox"})]})]}),t.jsxs("div",{children:[t.jsx("div",{style:{color:"var(--accent-color, #a855f7)",fontWeight:600,marginBottom:"8px"},children:"Selection"}),t.jsxs("div",{style:{display:"grid",gridTemplateColumns:"120px 1fr",gap:"6px 16px"},children:[t.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"Ctrl+Click"}),t.jsx("span",{children:"Toggle single item"}),t.jsx("kbd",{style:{background:"#333",padding:"2px 8px",borderRadius:"4px",fontSize:"0.85rem"},children:"Shift+Click"}),t.jsx("span",{children:"Select range"})]})]})]}),t.jsx("div",{style:{marginTop:"20px",paddingTop:"16px",borderTop:"1px solid var(--border-color, #333)",textAlign:"center"},children:t.jsxs("span",{style:{color:"var(--text-muted)",fontSize:"0.8rem"},children:["Press ",t.jsx("kbd",{style:{background:"#333",padding:"2px 6px",borderRadius:"4px"},children:"?"})," or ",t.jsx("kbd",{style:{background:"#333",padding:"2px 6px",borderRadius:"4px"},children:"Esc"})," to close"]})})]})}),R&&t.jsx("div",{style:{padding:"12px 16px",backgroundColor:"rgba(239, 68, 68, 0.1)",color:"#ef4444",textAlign:"center"},children:R}),b&&t.jsxs("div",{style:{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",color:"var(--text-muted)"},children:[t.jsx(_s,{size:40,style:{animation:"spin 1s linear infinite",marginBottom:"16px"}}),t.jsx("div",{children:"Loading media..."})]}),!b&&N.length===0&&t.jsxs("div",{style:{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",color:"var(--text-muted)"},children:[t.jsx("div",{style:{fontSize:"4rem",marginBottom:"16px",opacity:.5},children:"📁"}),t.jsxs("div",{style:{fontSize:"1.2rem",marginBottom:"8px"},children:["No ",c==="prompts"?"prompts":c==="all"?"media":c+"s"," yet"]}),t.jsx("div",{style:{fontSize:"0.9rem",opacity:.7},children:"Generated content will appear here"})]}),!b&&We.length>0&&c==="prompts"&&t.jsx("div",{ref:it,className:"prompts-list",onScroll:Kn,style:{flex:1,overflowY:"auto",overflowX:"hidden",padding:"16px",display:"flex",flexDirection:"column",gap:"12px"},children:We.slice(0,he).map((w,ce)=>t.jsxs("div",{style:{display:"flex",gap:"16px",padding:"16px",backgroundColor:"var(--bg-secondary, #1f1f1f)",borderRadius:"12px",border:"1px solid var(--border-color, #333)",cursor:"pointer",transition:"border-color 0.15s"},onClick:()=>C(ce),onMouseEnter:ae=>ae.currentTarget.style.borderColor="var(--accent-color, #a855f7)",onMouseLeave:ae=>ae.currentTarget.style.borderColor="var(--border-color, #333)",children:[t.jsx("div",{style:{flexShrink:0},children:w.type==="video"?t.jsx("video",{src:`${ve}${w.url}`,style:{width:"100px",height:"100px",objectFit:"cover",borderRadius:"8px"},autoPlay:!0,loop:!0,muted:!0,playsInline:!0}):t.jsx("img",{src:`${ve}${w.url}`,alt:w.filename,style:{width:"100px",height:"100px",objectFit:"cover",borderRadius:"8px"},loading:"lazy"})}),t.jsxs("div",{style:{flex:1,minWidth:0},children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:"8px"},children:[t.jsxs("div",{children:[t.jsx("div",{style:{fontSize:"0.85rem",fontWeight:600,color:"var(--text-primary)",marginBottom:"4px"},children:w.filename}),t.jsxs("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)"},children:[w.type==="video"?"🎬":w.type==="audio"?"🎵":"🖼️"," ",$t(w.size),w.metadata?.steps&&` • ${w.metadata.steps} steps`,w.metadata?.cfg&&` • CFG ${w.metadata.cfg}`]})]}),t.jsxs("div",{style:{display:"flex",gap:"8px"},children:[t.jsxs("button",{style:{background:"var(--accent-color, #a855f7)",border:"none",color:"#fff",padding:"6px 12px",borderRadius:"6px",cursor:"pointer",fontSize:"0.75rem",display:"flex",alignItems:"center",gap:"4px"},onClick:ae=>{ae.stopPropagation();const ke=w.metadata?.positive_prompt||w.metadata?.prompt;navigator.clipboard.writeText(ke)},children:[t.jsx(un,{size:12}),"Copy"]}),t.jsx("button",{className:(h.has(w.filename),""),style:{background:h.has(w.filename)?"#ef4444":"rgba(255,255,255,0.1)",border:"none",color:"#fff",padding:"6px",borderRadius:"6px",cursor:"pointer"},onClick:ae=>_t(w.filename,ae),children:t.jsx(Ur,{size:14,fill:h.has(w.filename)?"#fff":"none"})})]})]}),t.jsx("div",{style:{fontSize:"0.9rem",color:"var(--text-primary)",lineHeight:1.5,backgroundColor:"var(--bg-tertiary, #2a2a2a)",padding:"10px 12px",borderRadius:"6px",maxHeight:"100px",overflow:"hidden",textOverflow:"ellipsis",display:"-webkit-box",WebkitLineClamp:4,WebkitBoxOrient:"vertical"},children:w.metadata?.positive_prompt||w.metadata?.prompt})]})]},w.filename))}),!b&&We.length>0&&c!=="prompts"&&t.jsx("div",{ref:it,className:"media-grid",onScroll:Kn,style:{flex:1,overflowY:"auto",overflowX:"hidden",gridTemplateColumns:`repeat(${Y}, 1fr)`},children:We.slice(0,he).map((w,ce)=>t.jsxs("div",{className:`thumb-card ${L.has(ce)?"selected":""}`,style:{height:`${pe}px`},onClick:ae=>_n(ce,ae),children:[t.jsx("div",{className:"select-checkbox",onClick:ae=>pn(ce,ae),children:L.has(ce)&&t.jsx(mr,{size:14,color:"#fff"})}),t.jsx("div",{className:`favorite-btn ${h.has(w.filename)?"is-favorite":""}`,onClick:ae=>_t(w.filename,ae),title:h.has(w.filename)?"Remove from favorites":"Add to favorites",children:t.jsx(Ur,{size:14,color:h.has(w.filename)?"#fff":"rgba(255,255,255,0.7)",fill:h.has(w.filename)?"#fff":"none"})}),Je&&w.source==="storage"&&t.jsx("div",{className:`publish-btn ${q.has(`${fr(w.filename)}/${w.filename}`)?"is-published":""}`,onClick:ae=>{ae.stopPropagation(),$(w)},title:q.has(`${fr(w.filename)}/${w.filename}`)?"Published to gallery":"Publish to gallery",children:t.jsx(pt,{size:14,color:q.has(`${fr(w.filename)}/${w.filename}`)?"#fff":"rgba(255,255,255,0.7)",fill:q.has(`${fr(w.filename)}/${w.filename}`)?"#fff":"none"})}),(w.metadata?.positive_prompt||w.metadata?.prompt)&&t.jsx("button",{className:"prompt-bubble-btn",onClick:ae=>{ae.stopPropagation(),B({item:w})},title:"View prompt",children:"💬"}),w.has_source_image&&t.jsxs("div",{className:"source-image-badge",children:[t.jsx(Nn,{size:10}),t.jsx("span",{children:"+IMG"})]}),w.type==="video"?t.jsx("video",{src:`${ve}${w.url}`,autoPlay:!0,loop:!0,muted:!0,playsInline:!0,preload:"metadata",onLoadedMetadata:ae=>{const ke=ae.target.duration;ke&&!Re[w.filename]&&Ve(Ae=>({...Ae,[w.filename]:ke}))}}):w.type==="audio"?t.jsxs("div",{className:"audio-thumb",children:[t.jsx("div",{className:"audio-icon",children:"🎵"}),t.jsx("audio",{src:`${ve}${w.url}`,preload:"metadata",onLoadedMetadata:ae=>{const ke=ae.target.duration;ke&&!Re[w.filename]&&Ve(Ae=>({...Ae,[w.filename]:ke}))}})]}):t.jsx("img",{src:`${ve}${w.url}`,alt:w.filename,loading:"lazy"}),t.jsxs("div",{className:"media-overlay",children:[t.jsxs("div",{children:[t.jsx("div",{className:"media-filename",children:w.filename}),t.jsxs("div",{className:"media-size",children:[$t(w.size),(w.type==="video"||w.type==="audio")&&Re[w.filename]&&t.jsxs("span",{className:"media-duration",children:[t.jsx(Qn,{size:10}),ou(Re[w.filename])]})]})]}),t.jsxs("div",{className:"overlay-buttons",children:[w.metadata?.has_metadata&&t.jsx("button",{className:"overlay-btn",onClick:ae=>mn(w,ae),title:"Download metadata JSON",children:t.jsx(Ud,{size:14})}),t.jsx("button",{className:"overlay-btn",onClick:ae=>Mt(w,ae),title:"Download",children:t.jsx(qt,{size:14})})]})]})]},w.filename))}),Fe&&t.jsxs("div",{className:"lightbox-overlay",onClick:()=>C(null),children:[t.jsx("button",{className:"lightbox-close",onClick:()=>C(null),children:t.jsx(lt,{size:24})}),Fe.metadata?.has_metadata&&t.jsx("button",{style:{position:"absolute",top:"20px",left:"20px",padding:"8px 12px",borderRadius:"6px",background:K?"var(--accent-color, #a855f7)":"rgba(255,255,255,0.1)",border:"none",color:"#fff",cursor:"pointer",fontSize:"0.85rem",zIndex:1002},onClick:w=>{w.stopPropagation(),j(!K)},children:K?"Hide Prompt":"Show Prompt"}),K&&Fe.metadata&&t.jsxs("div",{className:"lightbox-metadata",onClick:w=>w.stopPropagation(),children:[Fe.metadata.positive_prompt&&t.jsxs("div",{style:{marginBottom:"16px"},children:[t.jsx("div",{className:"prompt-label",children:"✨ Positive Prompt"}),t.jsx("div",{className:"prompt-text",children:Fe.metadata.positive_prompt})]}),Fe.metadata.negative_prompt&&t.jsxs("div",{children:[t.jsx("div",{className:"prompt-label",children:"🚫 Negative Prompt"}),t.jsx("div",{className:"prompt-text",style:{color:"rgba(255,255,255,0.6)"},children:Fe.metadata.negative_prompt})]})]}),t.jsx("button",{className:"lightbox-nav",style:{left:"20px"},onClick:w=>{w.stopPropagation(),C(ce=>ce>0?ce-1:We.length-1)},children:t.jsx(Uf,{size:28})}),t.jsx("div",{className:"lightbox-content",onClick:w=>w.stopPropagation(),children:Fe.type==="video"?t.jsx("video",{src:`${ve}${Fe.url}`,autoPlay:!0,loop:!0,controls:!0,style:{borderRadius:"12px"}}):Fe.type==="audio"?t.jsxs("div",{className:"audio-lightbox",children:[t.jsx("div",{className:"audio-icon-large",children:"🎵"}),t.jsx("div",{className:"audio-filename",children:Fe.filename}),t.jsx("audio",{src:`${ve}${Fe.url}`,autoPlay:!0,controls:!0,style:{width:"100%",maxWidth:"400px",marginTop:"20px"}})]}):t.jsx("img",{src:`${ve}${Fe.url}`,alt:Fe.filename,style:{borderRadius:"12px"}})}),t.jsx("button",{className:"lightbox-nav",style:{right:"20px"},onClick:w=>{w.stopPropagation(),C(ce=>ce<We.length-1?ce+1:0)},children:t.jsx(Bf,{size:28})}),t.jsxs("div",{className:"lightbox-info",children:[t.jsx("span",{style:{color:"#fff",fontWeight:500},children:Fe.filename}),t.jsx("span",{style:{color:"rgba(255,255,255,0.6)"},children:$t(Fe.size)}),h.has(Fe.filename)&&t.jsx("span",{style:{color:"#ef4444",fontSize:"0.8rem"},children:"❤️ Favorite"}),Fe.has_source_image&&t.jsx("span",{style:{color:"#3b82f6",fontSize:"0.8rem"},children:"📷 Has source image"}),t.jsxs("span",{style:{color:"rgba(255,255,255,0.5)"},children:[A+1," / ",We.length]}),t.jsxs("div",{style:{display:"flex",gap:"8px"},children:[t.jsx("button",{className:"overlay-btn",onClick:w=>_t(Fe.filename,w),title:h.has(Fe.filename)?"Remove from favorites":"Add to favorites",style:{background:h.has(Fe.filename)?"rgba(239, 68, 68, 0.5)":void 0},children:t.jsx(Ur,{size:16,fill:h.has(Fe.filename)?"#ef4444":"none",color:h.has(Fe.filename)?"#ef4444":"#fff"})}),Fe.has_source_image&&Fe.source_image&&t.jsx("button",{className:"overlay-btn",onClick:w=>Mt(Fe.source_image,w),title:"Download source image",children:t.jsx(Nn,{size:16})}),Fe.metadata?.has_metadata&&t.jsx("button",{className:"overlay-btn",onClick:w=>mn(Fe,w),title:"Download metadata JSON",children:t.jsx(Ud,{size:16})}),t.jsx("button",{className:"overlay-btn",onClick:w=>Mt(Fe,w),title:"Download",children:t.jsx(qt,{size:16})})]})]})]}),k&&t.jsx("div",{className:"prompt-popup-overlay",onClick:()=>B(null),children:t.jsxs("div",{className:"prompt-popup",onClick:w=>w.stopPropagation(),children:[t.jsxs("div",{className:"prompt-popup-header",children:[t.jsxs("div",{className:"prompt-popup-title",children:[t.jsx(Pm,{size:18}),"Prompt Details"]}),t.jsx("button",{className:"prompt-popup-close",onClick:()=>B(null),children:t.jsx(lt,{size:20})})]}),t.jsxs("div",{className:"prompt-popup-content",children:[t.jsxs("div",{style:{display:"flex",gap:"12px",alignItems:"flex-start"},children:[k.item.type==="video"?t.jsx("video",{src:`${ve}${k.item.url}`,className:"prompt-media-preview",autoPlay:!0,loop:!0,muted:!0,playsInline:!0}):t.jsx("img",{src:`${ve}${k.item.url}`,alt:k.item.filename,className:"prompt-media-preview"}),t.jsxs("div",{style:{flex:1},children:[t.jsx("div",{style:{fontSize:"0.85rem",fontWeight:600,color:"var(--text-primary)"},children:k.item.filename}),t.jsxs("div",{style:{fontSize:"0.75rem",color:"var(--text-muted)",marginTop:"4px"},children:[k.item.type==="video"?"🎬 Video":"🖼️ Image"," • ",$t(k.item.size),k.item.type==="video"&&Re[k.item.filename]&&t.jsxs(t.Fragment,{children:[" • ",ou(Re[k.item.filename])]}),k.item.metadata?.width&&k.item.metadata?.height&&t.jsxs(t.Fragment,{children:[" • ",k.item.metadata.width,"×",k.item.metadata.height]})]})]})]}),(k.item.metadata?.positive_prompt||k.item.metadata?.prompt)&&t.jsxs("div",{className:"prompt-section",children:[t.jsx("div",{className:"prompt-section-label",children:"✨ Positive Prompt"}),t.jsx("div",{className:"prompt-section-text",children:k.item.metadata.positive_prompt||k.item.metadata.prompt}),t.jsxs("button",{className:"prompt-copy-btn",onClick:()=>{const w=k.item.metadata.positive_prompt||k.item.metadata.prompt;navigator.clipboard.writeText(w)},children:[t.jsx(un,{size:14}),"Copy Prompt"]})]}),k.item.metadata?.negative_prompt&&t.jsxs("div",{className:"prompt-section",children:[t.jsx("div",{className:"prompt-section-label",children:"🚫 Negative Prompt"}),t.jsx("div",{className:"prompt-section-text",style:{color:"var(--text-muted)"},children:k.item.metadata.negative_prompt})]}),(k.item.metadata?.steps||k.item.metadata?.cfg||k.item.metadata?.seed||k.item.metadata?.sampler||k.item.metadata?.model)&&t.jsxs("div",{className:"prompt-section",children:[t.jsx("div",{className:"prompt-section-label",children:"⚙️ Generation Settings"}),t.jsxs("div",{style:{display:"flex",gap:"12px",flexWrap:"wrap",fontSize:"0.85rem"},children:[k.item.metadata.steps&&t.jsxs("span",{children:["Steps: ",t.jsx("strong",{children:k.item.metadata.steps})]}),k.item.metadata.cfg&&t.jsxs("span",{children:["CFG: ",t.jsx("strong",{children:k.item.metadata.cfg})]}),k.item.metadata.seed&&t.jsxs("span",{children:["Seed: ",t.jsx("strong",{children:k.item.metadata.seed})]}),k.item.metadata.sampler&&t.jsxs("span",{children:["Sampler: ",t.jsx("strong",{children:k.item.metadata.sampler})]}),k.item.metadata.scheduler&&t.jsxs("span",{children:["Scheduler: ",t.jsx("strong",{children:k.item.metadata.scheduler})]})]}),k.item.metadata.model&&t.jsxs("div",{style:{marginTop:"8px",fontSize:"0.8rem",color:"var(--text-muted)"},children:["Model: ",t.jsx("strong",{style:{color:"var(--text-primary)"},children:k.item.metadata.model})]})]}),k.item.metadata?.loras&&k.item.metadata.loras.length>0&&t.jsxs("div",{className:"prompt-section",children:[t.jsx("div",{className:"prompt-section-label",children:"🎨 LoRAs Used"}),t.jsx("div",{style:{display:"flex",flexDirection:"column",gap:"6px",fontSize:"0.85rem"},children:k.item.metadata.loras.map((w,ce)=>t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"6px 10px",backgroundColor:"var(--bg-secondary)",borderRadius:"4px"},children:[t.jsx("span",{style:{fontFamily:"monospace",fontSize:"0.8rem",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:"80%"},children:w.name}),t.jsxs("span",{style:{color:"var(--accent-color, #a855f7)",fontWeight:600,fontSize:"0.8rem"},children:[(w.strength*100).toFixed(0),"%"]})]},ce))})]})]})]})}),m&&t.jsx(Kh,{mediaItem:m,onClose:()=>$(null),onPublished:w=>{le(ce=>new Set([...ce,w.storage_path])),$(null)}})]})}function tg({item:c,onClose:x}){const{user:d}=Ke(),[N,p]=l.useState(c.user_liked||!1),[b,S]=l.useState(c.like_count||0),[R,I]=l.useState(!1),[P,T]=l.useState(!1),[A,C]=l.useState(""),L=()=>`${ve}/user/media/${c.storage_path}`,V=async()=>{if(!d){C("Please log in to like items"),setTimeout(()=>C(""),3e3);return}try{C("");const Z=await rn(`/api/gallery/${c.id}/like`,{method:"POST"});if(!Z.ok)throw new Error("Failed to toggle like");const K=await Z.json();p(K.liked),S(K.like_count)}catch(Z){console.error("❌ Like error:",Z),C("Failed to update like status"),setTimeout(()=>C(""),3e3)}},U=async()=>{const Z=`${window.location.origin}/gallery/${c.id}`;try{await navigator.clipboard.writeText(Z),T(!0),setTimeout(()=>T(!1),2e3)}catch(K){console.error("Failed to copy:",K)}},D=async()=>{const Z=c.metadata?.positive_prompt||c.metadata?.prompt;if(Z)try{await navigator.clipboard.writeText(Z),I(!0),setTimeout(()=>I(!1),2e3)}catch(K){console.error("Failed to copy prompt:",K)}},ee=L();return t.jsx("div",{className:"modal-overlay",onClick:x,style:{position:"fixed",inset:0,background:"rgba(0,0,0,0.9)",zIndex:1e3,display:"flex",alignItems:"center",justifyContent:"center",padding:"20px"},children:t.jsxs("div",{className:"modal-content",onClick:Z=>Z.stopPropagation(),style:{maxWidth:"1200px",width:"100%",maxHeight:"90vh",background:"#1a1a1a",borderRadius:"12px",overflow:"hidden",display:"flex",flexDirection:"row",border:"1px solid #333"},children:[t.jsx("div",{style:{flex:"0 0 60%",background:"#000",display:"flex",alignItems:"center",justifyContent:"center",position:"relative"},children:c.media_type==="video"?t.jsx("video",{src:ee,controls:!0,autoPlay:!0,loop:!0,style:{maxWidth:"100%",maxHeight:"100%",objectFit:"contain"}}):t.jsx("img",{src:ee,alt:c.title,style:{maxWidth:"100%",maxHeight:"100%",objectFit:"contain"}})}),t.jsxs("div",{style:{flex:"0 0 40%",display:"flex",flexDirection:"column",overflow:"hidden"},children:[t.jsxs("div",{style:{padding:"20px",borderBottom:"1px solid #333",display:"flex",justifyContent:"space-between",alignItems:"flex-start"},children:[t.jsxs("div",{style:{flex:1},children:[t.jsx("h2",{style:{margin:"0 0 8px",fontSize:"1.25rem",fontWeight:600,color:"#fff"},children:c.title}),t.jsxs("div",{style:{display:"flex",gap:"16px",fontSize:"14px",color:"#888"},children:[t.jsxs("span",{style:{display:"flex",alignItems:"center",gap:"4px"},children:[t.jsx(uu,{size:16}),c.view_count," views"]}),t.jsxs("span",{style:{display:"flex",alignItems:"center",gap:"4px"},children:[t.jsx(Ur,{size:16}),b," likes"]})]})]}),t.jsx("button",{onClick:x,style:{background:"none",border:"none",color:"#ccc",cursor:"pointer",padding:"4px",display:"flex",marginLeft:"12px"},children:t.jsx(lt,{size:24})})]}),t.jsxs("div",{style:{flex:1,overflowY:"auto",padding:"20px"},children:[c.description&&t.jsx("div",{style:{marginBottom:"20px"},children:t.jsx("p",{style:{margin:0,fontSize:"14px",lineHeight:1.6,color:"#ccc"},children:c.description})}),c.tags&&c.tags.length>0&&t.jsxs("div",{style:{marginBottom:"20px"},children:[t.jsx("div",{style:{fontSize:"13px",color:"#888",marginBottom:"8px",fontWeight:500},children:"Tags"}),t.jsx("div",{style:{display:"flex",flexWrap:"wrap",gap:"6px"},children:c.tags.map((Z,K)=>t.jsx("span",{style:{fontSize:"12px",padding:"4px 10px",background:"#2a2a2a",borderRadius:"6px",color:"#aaa",border:"1px solid #333"},children:Z},K))})]}),(c.metadata?.positive_prompt||c.metadata?.prompt)&&t.jsxs("div",{style:{marginBottom:"20px"},children:[t.jsxs("div",{style:{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"8px"},children:[t.jsx("div",{style:{fontSize:"13px",color:"#888",fontWeight:500},children:"Prompt"}),t.jsx("button",{onClick:D,style:{padding:"4px 8px",background:R?"#10b981":"#2a2a2a",border:"1px solid #444",borderRadius:"4px",color:"#fff",fontSize:"12px",cursor:"pointer",display:"flex",alignItems:"center",gap:"4px"},children:R?t.jsxs(t.Fragment,{children:[t.jsx(mr,{size:12}),"Copied"]}):t.jsxs(t.Fragment,{children:[t.jsx(un,{size:12}),"Copy"]})})]}),t.jsx("div",{style:{padding:"12px",background:"#2a2a2a",borderRadius:"8px",fontSize:"13px",lineHeight:1.5,color:"#ccc",border:"1px solid #333",fontFamily:"monospace",whiteSpace:"pre-wrap",wordBreak:"break-word"},children:c.metadata?.positive_prompt||c.metadata?.prompt})]}),c.metadata?.negative_prompt&&t.jsxs("div",{style:{marginBottom:"20px"},children:[t.jsx("div",{style:{fontSize:"13px",color:"#888",marginBottom:"8px",fontWeight:500},children:"Negative Prompt"}),t.jsx("div",{style:{padding:"12px",background:"#2a2a2a",borderRadius:"8px",fontSize:"13px",lineHeight:1.5,color:"#ccc",border:"1px solid #333",fontFamily:"monospace",whiteSpace:"pre-wrap",wordBreak:"break-word"},children:c.metadata.negative_prompt})]}),c.metadata&&Object.keys(c.metadata).length>0&&t.jsxs("div",{style:{marginBottom:"20px"},children:[t.jsx("div",{style:{fontSize:"13px",color:"#888",marginBottom:"8px",fontWeight:500},children:"Settings"}),t.jsxs("div",{style:{padding:"12px",background:"#2a2a2a",borderRadius:"8px",fontSize:"12px",color:"#ccc",border:"1px solid #333"},children:[c.metadata.model&&t.jsxs("div",{style:{marginBottom:"6px"},children:[t.jsx("span",{style:{color:"#888"},children:"Model:"})," ",t.jsx("span",{style:{color:"#aaa"},children:c.metadata.model})]}),c.metadata.steps&&t.jsxs("div",{style:{marginBottom:"6px"},children:[t.jsx("span",{style:{color:"#888"},children:"Steps:"})," ",t.jsx("span",{style:{color:"#aaa"},children:c.metadata.steps})]}),c.metadata.cfg_scale&&t.jsxs("div",{style:{marginBottom:"6px"},children:[t.jsx("span",{style:{color:"#888"},children:"CFG Scale:"})," ",t.jsx("span",{style:{color:"#aaa"},children:c.metadata.cfg_scale})]}),c.metadata.seed&&t.jsxs("div",{children:[t.jsx("span",{style:{color:"#888"},children:"Seed:"})," ",t.jsx("span",{style:{color:"#aaa"},children:c.metadata.seed})]})]})]})]}),t.jsxs("div",{style:{padding:"16px 20px",borderTop:"1px solid #333",display:"flex",flexDirection:"column",gap:"12px"},children:[A&&t.jsxs("div",{style:{padding:"10px 12px",background:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.3)",borderRadius:"6px",color:"#ef4444",fontSize:"13px",display:"flex",alignItems:"center",gap:"8px"},children:[t.jsx(Yl,{size:16}),A]}),t.jsxs("div",{style:{display:"flex",gap:"12px"},children:[t.jsxs("button",{onClick:V,disabled:!d,style:{flex:1,padding:"10px 16px",background:N?"#ef4444":"#2a2a2a",border:"1px solid #444",borderRadius:"8px",color:"#fff",fontSize:"14px",fontWeight:500,cursor:d?"pointer":"not-allowed",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px",opacity:d?1:.5},children:[t.jsx(Ur,{size:16,fill:N?"#fff":"none"}),N?"Unlike":"Like"]}),t.jsx("button",{onClick:U,style:{flex:1,padding:"10px 16px",background:P?"#10b981":"#2a2a2a",border:"1px solid #444",borderRadius:"8px",color:"#fff",fontSize:"14px",fontWeight:500,cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center",gap:"8px"},children:P?t.jsxs(t.Fragment,{children:[t.jsx(mr,{size:16}),"Copied"]}):t.jsxs(t.Fragment,{children:[t.jsx(Jm,{size:16}),"Share"]})})]})]})]})]})})}function ng(){const{user:c}=Ke(),{nsfwEnabled:x}=Ga(),[d,N]=l.useState([]),[p,b]=l.useState(!1),[S,R]=l.useState(""),[I,P]=l.useState("all"),[T,A]=l.useState("created_at"),[C,L]=l.useState(1),[V,U]=l.useState(!0),[D,ee]=l.useState(0),Z=l.useRef(null),[K,j]=l.useState(null),k=l.useCallback(async(v=!1)=>{b(!0),R("");const te=v?1:C;v&&(L(1),N([]));try{const re=new URLSearchParams;I&&I!=="all"&&re.append("media_type",I),(!c||!x)&&re.append("is_nsfw","false"),re.append("sort_by",T),re.append("order","desc"),re.append("page",te.toString()),re.append("per_page","30");const xe=await rn(`/api/gallery?${re.toString()}`);if(!xe.ok)throw new Error("Failed to fetch gallery");const ge=await xe.json();console.log("📸 Gallery data:",ge),N(v?ge.items:E=>[...E,...ge.items]),ee(ge.total),U(ge.has_more)}catch(re){console.error("❌ Gallery error:",re),R(re.message||"Failed to load gallery")}finally{b(!1)}},[I,T,C,c,x]);l.useEffect(()=>{k(!0)},[I,T,c,x]);const B=l.useCallback(v=>{const{scrollTop:te,clientHeight:re,scrollHeight:xe}=v.target;xe-te-re<500&&!p&&V&&L(ge=>ge+1)},[p,V]);l.useEffect(()=>{C>1&&k(!1)},[C]);const h=v=>`${ve}/user/media/${v.storage_path}`;return t.jsxs("div",{style:{display:"flex",flexDirection:"column",height:"100%",background:"var(--bg-primary, #1a1a1a)",color:"var(--text-primary, #fff)"},children:[t.jsxs("div",{style:{padding:"16px 24px",borderBottom:"1px solid #333",display:"flex",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap",gap:"12px"},children:[t.jsxs("div",{children:[t.jsx("h2",{style:{margin:0,fontSize:"1.5rem",fontWeight:600},children:"🖼️ Community Gallery"}),t.jsxs("p",{style:{margin:"4px 0 0",fontSize:"0.9rem",color:"#888"},children:["Discover amazing AI creations from the community",D>0&&` · ${D} items`]})]}),t.jsxs("div",{style:{display:"flex",gap:"12px",alignItems:"center"},children:[t.jsxs("div",{style:{display:"flex",gap:"6px"},children:[t.jsxs("button",{onClick:()=>P("all"),style:{padding:"8px 12px",background:I==="all"?"#3b82f6":"#2a2a2a",border:"1px solid #444",borderRadius:"6px",color:"#fff",fontSize:"13px",cursor:"pointer",display:"flex",alignItems:"center",gap:"6px"},children:[t.jsx(pu,{size:14}),"All"]}),t.jsxs("button",{onClick:()=>P("video"),style:{padding:"8px 12px",background:I==="video"?"#3b82f6":"#2a2a2a",border:"1px solid #444",borderRadius:"6px",color:"#fff",fontSize:"13px",cursor:"pointer",display:"flex",alignItems:"center",gap:"6px"},children:[t.jsx(Xn,{size:14}),"Videos"]}),t.jsxs("button",{onClick:()=>P("image"),style:{padding:"8px 12px",background:I==="image"?"#3b82f6":"#2a2a2a",border:"1px solid #444",borderRadius:"6px",color:"#fff",fontSize:"13px",cursor:"pointer",display:"flex",alignItems:"center",gap:"6px"},children:[t.jsx(Nn,{size:14}),"Images"]})]}),t.jsxs("select",{value:T,onChange:v=>A(v.target.value),style:{padding:"8px 12px",background:"#2a2a2a",border:"1px solid #444",borderRadius:"6px",color:"#fff",fontSize:"13px",cursor:"pointer"},children:[t.jsx("option",{value:"created_at",children:"Newest"}),t.jsx("option",{value:"like_count",children:"Most Liked"}),t.jsx("option",{value:"view_count",children:"Most Viewed"})]}),t.jsx("button",{onClick:()=>k(!0),disabled:p,style:{padding:"8px 12px",background:"#2a2a2a",border:"1px solid #444",borderRadius:"6px",color:"#fff",cursor:p?"not-allowed":"pointer",display:"flex",alignItems:"center",opacity:p?.5:1},children:t.jsx(_s,{size:16,className:p?"spinning":""})})]})]}),!c&&t.jsx("div",{style:{margin:"16px 24px",padding:"12px 16px",background:"rgba(59, 130, 246, 0.1)",border:"1px solid rgba(59, 130, 246, 0.3)",borderRadius:"8px",fontSize:"14px",color:"#60a5fa"},children:"🔒 Log in to view all content. Gallery is filtered to SFW content for anonymous users."}),S&&t.jsxs("div",{style:{margin:"16px 24px",padding:"12px 16px",background:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.3)",borderRadius:"8px",fontSize:"14px",color:"#ef4444"},children:["❌ ",S]}),t.jsxs("div",{ref:Z,onScroll:B,style:{flex:1,overflowY:"auto",padding:"24px",display:"grid",gridTemplateColumns:"repeat(auto-fill, minmax(280px, 1fr))",gap:"20px",alignContent:"start"},children:[d.map(v=>t.jsxs("div",{onClick:()=>j(v),style:{background:"#2a2a2a",borderRadius:"12px",overflow:"hidden",cursor:"pointer",transition:"transform 0.2s, box-shadow 0.2s",border:"1px solid #333"},onMouseEnter:te=>{te.currentTarget.style.transform="translateY(-4px)",te.currentTarget.style.boxShadow="0 8px 24px rgba(0,0,0,0.4)"},onMouseLeave:te=>{te.currentTarget.style.transform="translateY(0)",te.currentTarget.style.boxShadow="none"},children:[t.jsxs("div",{style:{aspectRatio:"9/16",background:"#000",position:"relative",overflow:"hidden"},children:[v.media_type==="video"?t.jsx("video",{src:h(v),style:{width:"100%",height:"100%",objectFit:"cover"}}):t.jsx("img",{src:h(v),alt:v.title,style:{width:"100%",height:"100%",objectFit:"cover"}}),v.is_nsfw&&t.jsx("div",{style:{position:"absolute",top:"8px",right:"8px",background:"rgba(239, 68, 68, 0.9)",color:"#fff",padding:"4px 8px",borderRadius:"4px",fontSize:"11px",fontWeight:600},children:"🔞 NSFW"}),t.jsxs("div",{style:{position:"absolute",bottom:0,left:0,right:0,background:"linear-gradient(to top, rgba(0,0,0,0.8), transparent)",padding:"8px 12px",display:"flex",gap:"12px",fontSize:"12px",color:"#fff"},children:[t.jsxs("span",{style:{display:"flex",alignItems:"center",gap:"4px"},children:[t.jsx(uu,{size:14}),v.view_count]}),t.jsxs("span",{style:{display:"flex",alignItems:"center",gap:"4px"},children:[t.jsx(Ur,{size:14}),v.like_count]})]})]}),t.jsxs("div",{style:{padding:"12px"},children:[t.jsx("h3",{style:{margin:"0 0 6px",fontSize:"14px",fontWeight:600,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"},children:v.title}),v.description&&t.jsx("p",{style:{margin:"0 0 8px",fontSize:"12px",color:"#888",overflow:"hidden",textOverflow:"ellipsis",display:"-webkit-box",WebkitLineClamp:2,WebkitBoxOrient:"vertical",lineHeight:1.4},children:v.description}),v.tags&&v.tags.length>0&&t.jsxs("div",{style:{display:"flex",flexWrap:"wrap",gap:"4px",marginTop:"8px"},children:[v.tags.slice(0,3).map((te,re)=>t.jsx("span",{style:{fontSize:"11px",padding:"2px 8px",background:"#3a3a3a",borderRadius:"4px",color:"#aaa"},children:te},re)),v.tags.length>3&&t.jsxs("span",{style:{fontSize:"11px",color:"#666"},children:["+",v.tags.length-3]})]})]})]},v.id)),p&&t.jsx("div",{style:{gridColumn:"1 / -1",textAlign:"center",padding:"20px",color:"#888"},children:"Loading more..."}),!p&&!V&&d.length>0&&t.jsx("div",{style:{gridColumn:"1 / -1",textAlign:"center",padding:"20px",color:"#666"},children:"No more items"}),!p&&d.length===0&&t.jsxs("div",{style:{gridColumn:"1 / -1",textAlign:"center",padding:"60px 20px",color:"#666"},children:[t.jsx(Nn,{size:48,style:{marginBottom:"16px",opacity:.3}}),t.jsx("p",{style:{fontSize:"16px"},children:"No items in the gallery yet"}),t.jsx("p",{style:{fontSize:"14px",marginTop:"8px"},children:"Be the first to publish your creations!"})]})]}),K&&t.jsx(tg,{item:K,onClose:()=>j(null)})]})}const rg=()=>{const c=ve.startsWith("https")?"wss:":"ws:",x=ve.replace(/^https?:\/\//,"");return`${c}//${x}/ws/logs`};function sg(){const[c,x]=l.useState([]),[d,N]=l.useState(!0),[p,b]=l.useState(!1),[S,R]=l.useState(!1),I=l.useRef(null),P=l.useRef(null),T=l.useRef(null),A=l.useRef(0),C=5,L=3e3,V=l.useCallback(()=>{if(P.current?.readyState===WebSocket.OPEN||A.current>=C)return;const U=new WebSocket(rg());P.current=U,U.onopen=()=>{R(!0),A.current=0,console.log("📡 Log WebSocket connected")},U.onmessage=D=>{try{const ee=JSON.parse(D.data);x(Z=>[...Z,ee].slice(-500))}catch{}},U.onclose=()=>{if(R(!1),A.current++,A.current<C){const D=L*Math.pow(2,A.current-1);console.log(`📡 Log WebSocket disconnected, retry ${A.current}/${C} in ${D/1e3}s`),T.current=setTimeout(()=>{d&&V()},D)}else console.log("📡 Log WebSocket: max reconnect attempts reached, giving up")},U.onerror=()=>{U.close()}},[d]);return l.useEffect(()=>(d?V():(P.current?.close(),T.current&&clearTimeout(T.current)),()=>{P.current?.close(),T.current&&clearTimeout(T.current)}),[d,V]),l.useEffect(()=>{I.current&&I.current.scrollIntoView({behavior:"smooth"})},[c]),d?t.jsxs("div",{style:{position:"fixed",bottom:"20px",right:"20px",width:p?"800px":"400px",height:p?"600px":"300px",backgroundColor:"#0a0a0a",border:"1px solid #333",borderRadius:"8px",display:"flex",flexDirection:"column",zIndex:100,boxShadow:"0 10px 30px rgba(0,0,0,0.8)",transition:"all 0.2s ease"},children:[t.jsxs("div",{style:{padding:"8px 12px",borderBottom:"1px solid #333",display:"flex",justifyContent:"space-between",alignItems:"center",backgroundColor:"#121212",borderTopLeftRadius:"8px",borderTopRightRadius:"8px"},children:[t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"8px",fontSize:"0.8rem",fontWeight:600,color:"#a3a3a3"},children:[t.jsx(Bd,{size:14}),t.jsx("span",{children:"Server Logs"}),S?t.jsx(yx,{size:12,color:"#22c55e",title:"Connected"}):t.jsx(gx,{size:12,color:"#ef4444",title:"Disconnected"})]}),t.jsxs("div",{style:{display:"flex",gap:"8px"},children:[t.jsx("button",{onClick:()=>b(!p),style:{background:"transparent",border:"none",cursor:"pointer",color:"#666"},children:p?t.jsx(Lm,{size:14}):t.jsx(Em,{size:14})}),t.jsx("button",{onClick:()=>N(!1),style:{background:"transparent",border:"none",cursor:"pointer",color:"#666"},children:t.jsx(lt,{size:14})})]})]}),t.jsxs("div",{style:{flex:1,overflowY:"auto",padding:"12px",fontFamily:"monospace",fontSize:"0.75rem",color:"#d4d4d4",lineHeight:"1.4"},children:[c.map((U,D)=>t.jsxs("div",{style:{marginBottom:"4px",display:"flex",gap:"8px"},children:[t.jsx("span",{style:{color:"#525252",flexShrink:0},children:U.timestamp?.split("T")[1]?.split(".")[0]||""}),t.jsx("span",{style:{color:U.level==="ERROR"?"#ef4444":U.level==="WARNING"?"#eab308":"#a3a3a3"},children:U.message})]},D)),t.jsx("div",{ref:I})]})]}):t.jsx("button",{onClick:()=>N(!0),style:{position:"fixed",bottom:"20px",right:"20px",backgroundColor:"#1a1a1a",border:"1px solid #333",borderRadius:"50%",width:"48px",height:"48px",display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",zIndex:100,boxShadow:"0 4px 12px rgba(0,0,0,0.5)"},children:t.jsx(Bd,{size:20,color:"#a3a3a3"})})}function ag(){const[c,x]=l.useState(me.IMAGE_TO_VIDEO),[d,N]=l.useState(!1),{purchaseSuccess:p,purchaseCancelled:b,clearPurchaseSuccess:S,clearPurchaseCancelled:R}=Zl(),[I,P]=l.useState(null),[T,A]=l.useState(!1),[C,L]=l.useState(null),[V,U]=l.useState(0),[D,ee]=l.useState(0),[Z,K]=l.useState(!1),[j,k]=l.useState(null),[B,h]=l.useState(null),v=l.useRef(null),te=async()=>{try{const $=await(await fetch(`${ve}/health`)).json();P($)}catch{P(null)}};l.useEffect(()=>{te();const m=setInterval(te,1e4);return()=>clearInterval(m)},[]);const re=async()=>{if(!T&&window.confirm("Backend herstarten? Lopende jobs worden afgebroken.")){A(!0);try{await fetch(`${ve}/restart`,{method:"POST"}),await new Promise(m=>setTimeout(m,3e3)),await te()}catch(m){console.error("Restart failed:",m)}finally{A(!1)}}},xe=()=>{const m=v.current;if(!m){alert("Geen parameters beschikbaar");return}const $=new Blob([JSON.stringify(m,null,2)],{type:"application/json"}),q=URL.createObjectURL($),le=document.createElement("a");le.href=q,le.download=`${c}_params_${Date.now()}.json`,le.click(),URL.revokeObjectURL(q)},ge=l.useMemo(()=>{switch(c){case me.TEXT_TO_VIDEO:return"Text to Video";case me.IMAGE_TO_VIDEO:return"Image to Video";case me.TEXT_TO_IMAGE_TO_VIDEO:return"Text to Image to Video";case me.VIDEO_TO_VIDEO:return"Video to Video";case me.VIDEO_TO_TEXT:return"Video to Text";case me.VIDEO_UPSCALER:return"Video Upscaler";case me.FRAME_INTERPOLATION:return"Frame Interpolation";case me.PIPELINE:return"Pipeline";case me.LORA_TRAINING:return"LoRA Training";case me.TEXT_TO_IMAGE:return"Text to Image";case me.IMAGE_TO_IMAGE:return"Image to Image";case me.REFRAME:return"Reframe";case me.FACE_SWAP:return"Face Swap";case me.UPSCALER:return"Upscaler";case me.IMAGE_TO_TEXT:return"Image to Text";case me.PROMPT_GENERATOR:return"Prompt Generator";case me.AUDIO_GENERATION:return"Audio Generation";case me.VOICE_CLONING:return"Voice Cloning";case me.LIP_SYNC:return"Lip Sync";case me.SPEECH_TO_VIDEO:return"Speech to Video";case me.MY_MEDIA_ALL:return"My Media - All";case me.MY_MEDIA_VIDEOS:return"My Media - Videos";case me.MY_MEDIA_IMAGES:return"My Media - Images";case me.MY_MEDIA_PROMPTS:return"My Media - Prompts";case me.GALLERY:return"Community Gallery";default:return"Tool"}},[c]),E=()=>{const m=()=>U(F=>F+1),$=(F,_)=>{K(F),k(()=>_)},q=F=>{v.current=F},le=()=>{ee(F=>F+1)};switch(c){case me.TEXT_TO_VIDEO:return t.jsx(oh,{onOutput:L,onRefreshHistory:m,onParamsChange:q,onJobSubmitted:le});case me.IMAGE_TO_VIDEO:return t.jsx(fh,{onOutput:L,onRefreshHistory:m,onCreationsModeChange:$,onParamsChange:q,onJobSubmitted:le});case me.TEXT_TO_IMAGE_TO_VIDEO:return t.jsx(xh,{onOutput:L,onParamsChange:q,onJobSubmitted:le});case me.PIPELINE:return t.jsx(Eh,{});case me.LORA_TRAINING:return t.jsx(Ih,{onOutput:L});case me.MY_MEDIA_ALL:return t.jsx($r,{filter:"all"});case me.MY_MEDIA_VIDEOS:return t.jsx($r,{filter:"video"});case me.MY_MEDIA_IMAGES:return t.jsx($r,{filter:"image"});case me.MY_MEDIA_AUDIO:return t.jsx($r,{filter:"audio"});case me.MY_MEDIA_PROMPTS:return t.jsx($r,{filter:"prompts"});case me.GALLERY:return t.jsx(ng,{});case me.TEXT_TO_IMAGE:return t.jsx(mh,{onOutput:L,onJobSubmitted:le});case me.IMAGE_TO_TEXT:return t.jsx(Rh,{});case me.PROMPT_GENERATOR:return t.jsx(Mh,{});case me.IMAGE_TO_IMAGE:return t.jsx(Lh,{onOutput:L,onJobSubmitted:le});case me.UPSCALER:return t.jsx(Dh,{onOutput:L,onJobSubmitted:le});case me.VIDEO_TO_VIDEO:return t.jsx(vh,{onOutput:L,onJobSubmitted:le});case me.VIDEO_TO_TEXT:return t.jsx(wh,{});case me.VIDEO_UPSCALER:return t.jsx(_h,{onOutput:L,onJobSubmitted:le});case me.FRAME_INTERPOLATION:return t.jsx(zh,{onOutput:L,onJobSubmitted:le});case me.AUDIO_GENERATION:return t.jsx($h,{onOutput:L,onJobSubmitted:le});case me.VOICE_CLONING:return t.jsx(Vh,{onOutput:L,onJobSubmitted:le});case me.LIP_SYNC:return t.jsx(Gh,{onOutput:L,onJobSubmitted:le});case me.SPEECH_TO_VIDEO:return t.jsx(Ch,{onOutput:L,onJobSubmitted:le});case me.REFRAME:return t.jsx(qh,{onOutput:L,onJobSubmitted:le});case me.FACE_SWAP:return t.jsx(Yh,{onOutput:L,onJobSubmitted:le});default:return t.jsx(Xh,{title:ge})}},{nsfwEnabled:ue,setNsfwEnabled:fe}=Ga(),{user:ie,isAdult:W,showLoginModal:G,loginModalMessage:X,closeLoginModal:J}=Ke();return t.jsxs("div",{className:"dashboard-wrapper",children:[t.jsxs("div",{className:"dashboard-container",children:[t.jsx(Ix,{activeToolId:c,onSelectTool:x,collapsed:d,onToggleCollapsed:()=>N(m=>!m)}),t.jsxs("main",{className:"main-content",children:[t.jsxs("div",{className:"top-bar",children:[t.jsx("h1",{children:ge}),t.jsxs("div",{style:{display:"flex",alignItems:"center",gap:"12px"},children:[W&&t.jsx("button",{className:`nsfw-toggle ${ue?"nsfw-enabled":"nsfw-disabled"}`,onClick:()=>fe(!ue),title:ue?"NSFW content visible":"NSFW content hidden",children:ue?"🔞 NSFW":"🛡️ SFW"}),t.jsx(Mx,{refreshToken:D,onJobComplete:m=>{U($=>$+1),m.output_video&&L({kind:"video",url:`${ve}${m.output_video}`,backendUrl:`${ve}${m.output_video}`})}}),t.jsx("button",{className:"icon-btn",onClick:re,disabled:T,title:"Herstart Backend",style:{opacity:T?.5:1,fontSize:"16px"},children:T?"⏳":"🔄"}),t.jsxs("div",{className:"status-indicator",children:[t.jsx("div",{className:`status-dot ${I?.status==="healthy"?"connected":""}`}),t.jsx("span",{children:I?.status==="healthy"?"Connected":"Disconnected"})]}),t.jsx(Bx,{})]})]}),p&&t.jsxs("div",{style:{background:"linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1))",border:"1px solid rgba(16, 185, 129, 0.3)",borderRadius:"8px",padding:"12px 16px",margin:"0 16px 16px",display:"flex",alignItems:"center",gap:"12px",fontSize:"0.9rem",color:"#10b981"},children:[t.jsx(du,{size:20}),t.jsxs("span",{style:{flex:1},children:[t.jsx("strong",{children:"Credits purchased successfully!"})," Your balance has been updated."]}),t.jsx("button",{onClick:S,style:{background:"none",border:"none",color:"#10b981",cursor:"pointer",padding:"4px 8px",fontSize:"1.2rem"},children:"×"})]}),b&&t.jsxs("div",{style:{background:"rgba(239, 68, 68, 0.1)",border:"1px solid rgba(239, 68, 68, 0.3)",borderRadius:"8px",padding:"12px 16px",margin:"0 16px 16px",display:"flex",alignItems:"center",gap:"12px",fontSize:"0.9rem",color:"#ef4444"},children:[t.jsx(Zf,{size:20}),t.jsx("span",{style:{flex:1},children:"Purchase cancelled. No charges were made."}),t.jsx("button",{onClick:R,style:{background:"none",border:"none",color:"#ef4444",cursor:"pointer",padding:"4px 8px",fontSize:"1.2rem"},children:"×"})]}),c===me.MY_MEDIA_ALL||c===me.MY_MEDIA_VIDEOS||c===me.MY_MEDIA_IMAGES||c===me.MY_MEDIA_AUDIO||c===me.MY_MEDIA_PROMPTS||c===me.GALLERY?t.jsx("div",{style:{flex:1,display:"flex",flexDirection:"column",overflow:"hidden"},children:E()}):t.jsxs("div",{className:"workspace",children:[t.jsxs("section",{className:"controls-panel",children:[t.jsxs("div",{className:"panel-header",style:{marginBottom:"16px",display:"flex",justifyContent:"space-between",alignItems:"center"},children:[t.jsx("div",{className:"panel-title",style:{fontSize:"0.85rem",fontWeight:600,color:"var(--text-secondary)",textTransform:"uppercase",letterSpacing:"0.05em"},children:"Parameters"}),t.jsx("button",{className:"icon-btn",onClick:xe,title:"Download parameters als JSON",style:{padding:"4px"},children:t.jsx(qt,{size:16})})]}),t.jsx("div",{className:"panel-body",children:E()})]}),C?t.jsx(Tx,{output:C,refreshToken:V,onSelectHistoryVideo:L,onClose:()=>L(null)}):t.jsxs("section",{className:"output-panel",style:{display:"flex",flexDirection:"column"},children:[Z&&t.jsxs("div",{style:{padding:"12px 16px",borderBottom:"1px solid var(--border-color)",backgroundColor:"var(--bg-secondary)",display:"flex",justifyContent:"space-between",alignItems:"center"},children:[t.jsx("span",{style:{fontWeight:600,color:"var(--text-primary)"},children:"Select Image for I2V"}),t.jsx("span",{style:{fontSize:"0.8rem",color:"var(--text-muted)"},children:"Click an image to use it"})]}),t.jsx("div",{style:{flex:1,overflow:"hidden"},children:t.jsx($r,{filter:"all",selectionMode:Z,onSelectItem:j})})]})]})]})]}),t.jsx(Wx,{onShowLegal:h}),B&&t.jsx(Yx,{type:B,onClose:()=>h(null)}),G&&t.jsx(Xx,{message:X,onClose:J}),ie?.email==="mark.op.mobiel@gmail.com"&&t.jsx(sg,{})]})}function og(){const{loading:c}=Ke();return c?t.jsx("div",{className:"app-loading",children:t.jsx("div",{className:"app-loading-spinner"})}):t.jsx($x,{children:t.jsx(Kx,{children:t.jsx(ag,{})})})}function lg(){return t.jsx(Lx,{children:t.jsx(og,{})})}If.createRoot(document.getElementById("root")).render(t.jsx(kf.StrictMode,{children:t.jsx(lg,{})}));
