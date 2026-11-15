/* ssalmuck-ai: /keys (design.md 6.20, 6.23). One Google check, then a passkey and a key shown once; later a passkey
   login for usage, rotate, revoke and logout. Talks only to this origin's /keys/api/* (POST, X-Requested-With: sclmd,
   same-origin credentials). The key is never written to storage, a URL or the console. */
(function () {
  "use strict";

  var API = "/keys/api/";

  // ---------- base64url and the WebAuthn JSON forms (also run under node: tests/site_keys_check.js) ----------

  function b64uToBytes(s) {
    if (typeof s !== "string" || !/^[A-Za-z0-9_-]*={0,2}$/.test(s)) throw new TypeError("not base64url");
    s = s.replace(/=+$/, "").replace(/-/g, "+").replace(/_/g, "/");
    while (s.length % 4) s += "=";
    var bin = atob(s), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function bytesToB64u(buf) {
    if (buf === null || buf === undefined) return null;
    var u8 = buf instanceof ArrayBuffer ? new Uint8Array(buf) : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    var parts = [];
    for (var i = 0; i < u8.length; i += 0x8000) parts.push(String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)));
    return btoa(parts.join("")).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  function copyJSON(o) { return JSON.parse(JSON.stringify(o)); }

  function unwrap(json) {
    if (!json || typeof json !== "object") throw new TypeError("no options");
    return json.publicKey && typeof json.publicKey === "object" ? json.publicKey : json;
  }

  function hasNative(name) {
    return typeof PublicKeyCredential !== "undefined" && typeof PublicKeyCredential[name] === "function";
  }

  function creationOptions(json, noNative) {
    var o = unwrap(json);
    if (!noNative && hasNative("parseCreationOptionsFromJSON")) {
      try { return PublicKeyCredential.parseCreationOptionsFromJSON(o); } catch (e) { /* an older parser: by hand below */ }
    }
    var p = copyJSON(o);
    p.challenge = b64uToBytes(o.challenge);
    if (!p.user || typeof o.user.id !== "string") throw new TypeError("no user id");
    p.user.id = b64uToBytes(o.user.id);
    if (Array.isArray(o.excludeCredentials)) {
      p.excludeCredentials = o.excludeCredentials.map(function (c) { var d = copyJSON(c); d.id = b64uToBytes(c.id); return d; });
    }
    return p;
  }

  function requestOptions(json, noNative) {
    var o = unwrap(json);
    if (!noNative && hasNative("parseRequestOptionsFromJSON")) {
      try { return PublicKeyCredential.parseRequestOptionsFromJSON(o); } catch (e) { /* by hand below */ }
    }
    var p = copyJSON(o);
    p.challenge = b64uToBytes(o.challenge);
    if (Array.isArray(o.allowCredentials)) {
      p.allowCredentials = o.allowCredentials.map(function (c) { var d = copyJSON(c); d.id = b64uToBytes(c.id); return d; });
    }
    return p;
  }

  function call(obj, name) {
    try { return typeof obj[name] === "function" ? obj[name]() : undefined; } catch (e) { return undefined; }
  }

  function credentialJSON(cred, noNative) {
    if (!noNative && typeof cred.toJSON === "function") {
      try {
        var j = cred.toJSON();
        if (j && typeof j === "object" && j.response) return j;
      } catch (e) { /* some password managers' credentials: by hand below */ }
    }
    var r = cred.response, out = {
      id: cred.id,
      rawId: bytesToB64u(cred.rawId),
      type: cred.type,
      response: { clientDataJSON: bytesToB64u(r.clientDataJSON) },
      clientExtensionResults: call(cred, "getClientExtensionResults") || {}
    };
    if (cred.authenticatorAttachment) out.authenticatorAttachment = cred.authenticatorAttachment;
    if (r.attestationObject) {
      out.response.attestationObject = bytesToB64u(r.attestationObject);
      out.response.transports = call(r, "getTransports") || [];
      var alg = call(r, "getPublicKeyAlgorithm"), pub = call(r, "getPublicKey"), data = call(r, "getAuthenticatorData");
      if (typeof alg === "number") out.response.publicKeyAlgorithm = alg;
      if (pub) out.response.publicKey = bytesToB64u(pub);
      if (data) out.response.authenticatorData = bytesToB64u(data);
    } else {
      out.response.authenticatorData = bytesToB64u(r.authenticatorData);
      out.response.signature = bytesToB64u(r.signature);
      out.response.userHandle = r.userHandle ? bytesToB64u(r.userHandle) : null;
    }
    return out;
  }

  // ---------- what to tell the person ----------

  var CODES = {};
  function say(codes, text) { codes.split(" ").forEach(function (c) { CODES[c] = text; }); }
  // from /keys/callback/google (#error=<code>)
  say("denied access_denied consent_denied cancelled", "구글 확인을 취소했습니다. 키를 받으려면 '구글로 확인'을 다시 누르세요.");
  say("state_invalid invalid_state state_mismatch bad_state state_unknown", "확인 요청이 맞지 않습니다. 다른 탭에서 다시 시작했거나 오래된 주소입니다. 처음부터 다시 하세요.");
  say("state_expired expired login_expired", "구글 확인을 시작한 지 10분이 지났습니다. 처음부터 다시 하세요.");
  say("token token_invalid id_token_invalid invalid_token nonce_mismatch invalid_id_token", "구글의 확인 결과를 검증하지 못했습니다. 다시 하세요. 계속되면 문의해 주세요.");
  say("google google_error google_unavailable provider_error exchange_failed upstream_error token_exchange_failed", "구글과 통신하지 못했습니다. 잠시 후 다시 하세요.");
  say("missing_code missing_state", "구글에서 돌아온 주소가 완전하지 않습니다. 처음부터 다시 하세요.");
  // limits on the key endpoints (6.20: keys per IP per day 3)
  say("too_many_keys key_limit keys_per_ip ip_daily_limit daily_key_limit", "이 IP 주소에서 하루에 받을 수 있는 키(3개)를 모두 받았습니다.");
  say("rate_limited rate_limit_exceeded", "요청이 너무 잦거나, 이 IP 주소에서 하루에 받을 수 있는 키(3개)를 모두 받았습니다.");
  say("too_many_requests", "요청이 너무 잦습니다. 잠시 후 다시 하세요.");
  say("busy", "키 발급 요청이 많습니다. 잠시 후 다시 하세요.");
  say("browser", "구글 확인을 시작한 브라우저와 돌아온 브라우저가 다르거나 쿠키가 막혀 있습니다. 같은 브라우저에서 처음부터 다시 하세요.");
  // sessions and CSRF
  say("google_check_needed not_verified verification_required google_required verify_first", "먼저 '구글로 확인'을 하세요. 확인은 15분 동안 유효합니다.");
  say("session_required no_session session_expired not_logged_in unauthorized login_required invalid_session", "로그인이 필요합니다. '패스키로 로그인'을 누르세요.");
  say("origin_mismatch bad_origin csrf csrf_failed forbidden missing_header", "이 사이트에서 온 요청으로 확인되지 않았습니다. 주소가 https://ssalmuck-ai.duckdns.org/keys 인지 확인하고 새로 고침하세요.");
  // passkeys
  say("challenge_expired challenge_missing challenge_invalid unknown_challenge", "패스키 요청의 유효 시간이 지났습니다. 버튼을 다시 누르세요.");
  say("credential_invalid invalid_credential passkey_invalid verification_failed attestation_invalid assertion_invalid signature_invalid bad_credential origin_invalid rp_mismatch", "패스키를 확인하지 못했습니다. 다시 하세요. 회수한 키의 패스키는 쓸 수 없으므로 구글로 다시 확인해 새 키를 받으세요.");
  say("credential_unknown unknown_credential passkey_unknown credential_not_found passkey_not_found", "등록되지 않은 패스키입니다. 키를 회수했거나 다른 사이트의 패스키입니다. 구글로 다시 확인하면 새 키를 받을 수 있습니다.");
  say("credential_exists already_registered", "이미 등록된 패스키입니다. '패스키로 로그인'을 쓰세요.");
  say("key_not_found no_key revoked key_revoked account_not_found", "이 계정에는 쓸 수 있는 키가 없습니다. 구글로 확인해 새 키를 받으세요.");
  // the rest
  say("bad_body bad_request invalid_request invalid_request_error invalid_json method_not_allowed", "요청 형식이 맞지 않습니다. 새로 고침한 뒤 다시 하세요.");
  say("server server_error internal_error", "서버 오류입니다. 잠시 후 다시 하세요.");
  say("unavailable service_unavailable keys_disabled not_configured starting", "키 발급이 꺼져 있거나 서버가 다시 시작하는 중입니다. 잠시 후 다시 하세요.");
  say("not_found unknown_path", "서버에서 키 발급 기능을 찾지 못했습니다. 아직 켜지 않았을 수 있습니다.");
  say("unknown", "알 수 없는 오류입니다. 처음부터 다시 하세요.");

  var STATUS = {
    400: "요청 형식이 맞지 않습니다. 새로 고침한 뒤 다시 하세요.",
    401: CODES.session_required,
    403: "서버가 요청을 거절했습니다. 새로 고침한 뒤 다시 하세요.",
    404: CODES.not_found,
    405: "서버가 이 요청 방식을 받지 않습니다. 새로 고침한 뒤 다시 하세요.",
    413: "요청이 너무 큽니다.",
    415: "요청 형식이 맞지 않습니다. 새로 고침한 뒤 다시 하세요.",
    429: CODES.rate_limited,
    500: CODES.server_error,
    502: "서버가 응답하지 않습니다. GPU PC가 꺼져 있거나 다시 시작하는 중일 수 있습니다. 잠시 후 다시 하세요.",
    503: CODES.unavailable,
    504: "서버 응답이 너무 늦습니다. 잠시 후 다시 하세요."
  };

  var WEBAUTHN = {
    NotAllowedError: "패스키 창이 닫혔거나 시간이 지났습니다. 로그인 중이라면 이 기기에 이 사이트의 패스키가 없을 수 있습니다. 다시 하세요.",
    InvalidStateError: "이 인증기에는 이미 이 계정의 패스키가 있습니다. '패스키로 로그인'을 쓰세요.",
    SecurityError: "이 주소에서는 패스키를 쓸 수 없습니다. https://ssalmuck-ai.duckdns.org/keys 에서 여세요.",
    NotSupportedError: "이 브라우저나 인증기는 필요한 패스키 방식을 지원하지 않습니다. 최신 Chrome, Edge, Safari, Firefox를 쓰세요.",
    AbortError: "패스키 요청이 중단되었습니다. 다시 하세요.",
    ConstraintError: "이 인증기는 기기에 저장되는 패스키를 만들 수 없습니다. 다른 인증기를 쓰세요.",
    TypeError: "서버의 패스키 요청을 이 브라우저가 읽지 못했습니다. 새로 고침한 뒤 다시 하세요.",
    UnknownError: "인증기에서 알 수 없는 오류가 났습니다. 다시 하세요."
  };

  var TIMEOUT_MS = 10000;  // every fetch of this page

  function later(seconds) {
    var s = Number(seconds);
    if (!isFinite(s) || s <= 0) return "";
    if (s < 90) return " 약 " + Math.ceil(s) + "초 뒤에 다시 할 수 있습니다.";
    if (s < 5400) return " 약 " + Math.ceil(s / 60) + "분 뒤에 다시 할 수 있습니다.";
    return " 약 " + Math.ceil(s / 3600) + "시간 뒤에 다시 할 수 있습니다.";
  }

  function safeCode(c) { return typeof c === "string" && /^[a-z0-9_.-]{1,48}$/.test(c) ? c : ""; }

  // {text, detail}: detail is shown small (the code and status), never the server's free text
  function messageFor(err) {
    if (!err) return { text: CODES.unknown, detail: "" };
    if (err.kind === "webauthn") return { text: WEBAUTHN[err.name] || WEBAUTHN.UnknownError, detail: "패스키 오류: " + (WEBAUTHN[err.name] ? err.name : "UnknownError") };
    if (err.kind === "network") return { text: "서버에 연결하지 못했습니다. 인터넷 연결을 확인하고 다시 하세요.", detail: "" };
    if (err.kind === "timeout") return { text: "서버가 10초 안에 응답하지 않았습니다. 잠시 후 다시 하세요.", detail: "" };
    if (err.kind === "unsupported") return { text: "이 브라우저는 패스키를 지원하지 않습니다. 최신 Chrome, Edge, Safari, Firefox에서 여세요.", detail: "" };
    var code = safeCode(err.code), status = err.status | 0;
    var text = (code && CODES[code]) || STATUS[status] || (status >= 500 ? CODES.server_error : CODES.unknown);
    if (status === 429 || code === "rate_limited" || code === "rate_limit_exceeded" || code === "too_many_keys") text += later(err.retryAfter) || (CODES[code] === CODES.too_many_keys || CODES[code] === CODES.rate_limited ? " 시간이 지나면 다시 할 수 있습니다." : "");
    var detail = [code ? "오류 코드 " + code : "", status ? "HTTP " + status : ""].filter(Boolean).join(", ");
    return { text: text, detail: detail };
  }

  // the fragment /keys/callback/google leaves: #verified or #error=<code>
  function callbackState(hash) {
    var h = String(hash || "").replace(/^#/, "");
    if (h === "verified") return { verified: true };
    if (h.indexOf("error=") === 0) {
      var c = h.slice(6);
      try { c = decodeURIComponent(c); } catch (e) { c = ""; }
      return { error: safeCode(c) || "unknown" };
    }
    return null;
  }

  // ---------- the API ----------

  function ApiError(status, code, retryAfter) { this.kind = "api"; this.status = status; this.code = code; this.retryAfter = retryAfter; }

  // one POST to /keys/api/<path>; the whole exchange (headers and body) must end within TIMEOUT_MS, or it is
  // aborted and fails as {kind: "timeout"}
  function post(path, body) {
    var ctl = typeof AbortController === "function" ? new AbortController() : null;
    var init = {
      method: "POST", mode: "same-origin", credentials: "same-origin", cache: "no-store", redirect: "error",
      headers: { "Content-Type": "application/json", "Accept": "application/json", "X-Requested-With": "sclmd" },
      body: JSON.stringify(body || {})
    };
    if (ctl) init.signal = ctl.signal;
    return new Promise(function (resolve, reject) {
      var over = false;
      var timer = setTimeout(function () {
        over = true;
        if (ctl) ctl.abort();
        reject({ kind: "timeout" });
      }, TIMEOUT_MS);
      function done(fn, v) { clearTimeout(timer); fn(v); }
      fetch(API + path, init).then(function (r) {
        return r.text().then(function (t) { return { r: r, t: t }; });
      }).then(function (x) {
        var r = x.r, j = null;
        try { j = x.t ? JSON.parse(x.t) : null; } catch (e) { j = null; }
        if (!r.ok) {
          var e = j && j.error && typeof j.error === "object" ? j.error : {};
          return done(reject, new ApiError(r.status, typeof e.code === "string" ? e.code : "", r.headers.get("Retry-After")));
        }
        if (!j || typeof j !== "object") return done(reject, new ApiError(r.status, "invalid_response", null));
        done(resolve, j);
      }, function () {
        done(reject, over ? { kind: "timeout" } : { kind: "network" });
      });
    });
  }

  function webauthn(fn) {
    return Promise.resolve().then(fn).then(function (c) {
      if (!c) throw { kind: "webauthn", name: "NotAllowedError" };
      return c;
    }, function (e) {
      throw { kind: "webauthn", name: e && typeof e.name === "string" ? e.name : "UnknownError" };
    });
  }

  function supported() {
    return typeof window !== "undefined" && typeof window.PublicKeyCredential === "function" && !!navigator.credentials &&
      typeof navigator.credentials.create === "function" && typeof navigator.credentials.get === "function";
  }

  window.SclmdKeys = {
    b64uToBytes: b64uToBytes, bytesToB64u: bytesToB64u, creationOptions: creationOptions, requestOptions: requestOptions,
    credentialJSON: credentialJSON, messageFor: messageFor, callbackState: callbackState, CODES: CODES, STATUS: STATUS,
    WEBAUTHN: WEBAUTHN
  };

  if (typeof document === "undefined" || !document.getElementById || !document.getElementById("keys-app")) return;

  // ---------- the page ----------

  var $ = function (id) { return document.getElementById(id); };
  var sections = ["s-start", "s-register", "s-key", "s-manage"];
  var keyOnScreen = false;

  var acting = false;  // a button was pressed: move the focus to what it opened (not on the page's first load)

  function show(id) {
    sections.forEach(function (s) { $(s).hidden = s !== id; });
    var h = $(id).querySelector("h2");
    if (h && acting) { h.setAttribute("tabindex", "-1"); h.focus(); }
  }

  function clearMessage() { $("msg").hidden = true; $("ok").hidden = true; }

  function fail(err) {
    var m = messageFor(err);
    $("msg-text").textContent = m.text;
    $("msg-detail").textContent = m.detail;
    $("msg-detail").hidden = !m.detail;
    $("ok").hidden = true;
    $("msg").hidden = false;
    $("msg").focus();
  }

  function tell(text) {
    $("ok-text").textContent = text;
    $("msg").hidden = true;
    $("ok").hidden = false;
  }

  function busy(button, on, label) {
    if (!button) return;
    if (on) { button.setAttribute("data-label", button.textContent); button.textContent = label || "기다리는 중…"; button.disabled = true; }
    else { button.textContent = button.getAttribute("data-label") || button.textContent; button.disabled = false; }
  }

  function run(button, label, work) {
    clearMessage();
    busy(button, true, label);
    acting = !!button;
    return Promise.resolve().then(work).then(function () { busy(button, false); }, function (e) { busy(button, false); fail(e); });
  }

  function showKey(key, note) {
    if (typeof key !== "string" || !/^[\x21-\x7e]{8,512}$/.test(key)) throw new ApiError(0, "invalid_response", null);
    $("key-value").textContent = key;
    $("key-note").textContent = note || "";
    $("key-note").hidden = !note;
    keyOnScreen = true;
    show("s-key");
  }

  function clearKey() {
    $("key-value").textContent = "";
    keyOnScreen = false;
  }

  function when(v) {
    if (v === null || v === undefined || v === "") return "?";
    var d = typeof v === "number" ? new Date(v < 1e12 ? v * 1000 : v) : new Date(String(v));
    return isNaN(d.getTime()) ? String(v) : d.toLocaleString("ko-KR", { dateStyle: "long", timeStyle: "short" });
  }

  function num(v) { return typeof v === "number" && isFinite(v) ? Math.round(v).toLocaleString("ko-KR") : "-"; }

  function usageRow(label, part) {
    var tr = document.createElement("tr"), th = document.createElement("th");
    th.textContent = label;
    tr.appendChild(th);
    var p = part || {};
    [num(p.tokens) + " / " + num(p.limit_tokens), num(p.requests) + (p.limit_requests ? " / " + num(p.limit_requests) : "")].forEach(function (t) {
      var td = document.createElement("td");
      td.className = "num";
      td.textContent = t;
      tr.appendChild(td);
    });
    return tr;
  }

  function showManage(me) {
    $("me-name").textContent = typeof me.name === "string" ? me.name : "?";
    $("me-created").textContent = when(me.created);
    $("me-tier").textContent = me.tier === "key" ? "키" : String(me.tier || "?");
    var body = $("me-usage");
    while (body.firstChild) body.removeChild(body.firstChild);
    var u = me.usage && typeof me.usage === "object" ? me.usage : {};
    body.appendChild(usageRow("지난 1시간", u.hour));
    body.appendChild(usageRow("지난 24시간", u.day));
    var over = $("me-over");
    over.hidden = !u.over_quota;
    over.textContent = u.over_quota ? "시간당 한도를 넘었습니다. 요청은 받지만 처리 순서가 크게 밀립니다." +
      (u.back_in_quota_s ? " 약 " + Math.ceil(u.back_in_quota_s / 60) + "분 후 한도 안으로 돌아옵니다." : "") : "";
    show("s-manage");
  }

  function loadMe(quiet) {
    return post("me").then(showManage, function (e) {
      if (quiet && e && (e.status === 401 || e.status === 403 || e.status === 404)) { show("s-start"); return; }
      show("s-start");
      throw e;
    });
  }

  function register() {
    if (!supported()) return Promise.reject({ kind: "unsupported" });
    return post("passkey/register/options").then(function (opts) {
      var pk;
      try { pk = creationOptions(opts); } catch (e) { throw { kind: "webauthn", name: "TypeError" }; }
      return webauthn(function () { return navigator.credentials.create({ publicKey: pk }); });
    }).then(function (cred) {
      return post("passkey/register/verify", { credential: credentialJSON(cred) });
    }).then(function (res) {
      showKey(res.key, res.replaced ? "이 구글 계정의 이전 키는 회수되었습니다. 이전 키를 쓰던 곳을 이 키로 바꾸세요." : "");
    });
  }

  function login() {
    if (!supported()) return Promise.reject({ kind: "unsupported" });
    return post("passkey/login/options").then(function (opts) {
      var pk;
      try { pk = requestOptions(opts); } catch (e) { throw { kind: "webauthn", name: "TypeError" }; }
      return webauthn(function () { return navigator.credentials.get({ publicKey: pk }); });
    }).then(function (cred) {
      return post("passkey/login/verify", { credential: credentialJSON(cred) });
    }).then(function () { return loadMe(false); });
  }

  function rotate() {
    if (!window.confirm("키를 재발급할까요? 지금 키는 바로 회수됩니다.")) return Promise.resolve();
    return post("rotate").then(function (res) { showKey(res.key, "이전 키는 회수되었습니다. 이전 키를 쓰던 곳을 이 키로 바꾸세요."); });
  }

  function revoke() {
    if (!window.confirm("키를 회수할까요? 키, 패스키 기록, 구글 계정의 HMAC이 서버에서 삭제됩니다. 같은 구글 계정으로 새 키를 다시 받을 수 있습니다.")) return Promise.resolve();
    return post("revoke").then(function () {
      show("s-start");
      tell("키를 회수했습니다. 기기에 남은 패스키는 기기 설정에서 지울 수 있습니다.");
    });
  }

  function logout() {
    return post("logout").then(function () { show("s-start"); tell("로그아웃했습니다."); });
  }

  function init() {
    if (!supported()) {
      $("unsupported").hidden = false;
      $("register").disabled = true;
      $("login").disabled = true;
    }
    $("register").addEventListener("click", function (ev) { run(ev.currentTarget, "패스키 창을 기다리는 중…", register); });
    $("login").addEventListener("click", function (ev) { run(ev.currentTarget, "패스키 창을 기다리는 중…", login); });
    $("rotate").addEventListener("click", function (ev) { run(ev.currentTarget, "바꾸는 중…", rotate); });
    $("revoke").addEventListener("click", function (ev) { run(ev.currentTarget, "회수하는 중…", revoke); });
    $("logout").addEventListener("click", function (ev) { run(ev.currentTarget, "로그아웃하는 중…", logout); });
    $("copy-key").addEventListener("click", function (ev) { window.SclmdSite.copyText($("key-value").textContent, ev.currentTarget); });
    $("key-done").addEventListener("click", function (ev) {
      clearKey();
      run(ev.currentTarget, "확인하는 중…", function () {
        return loadMe(true).then(function () {
          if ($("s-manage").hidden) tell("키를 화면에서 지웠습니다. 키를 관리하려면 '패스키로 로그인'을 누르세요.");
        });
      });
    });
    window.addEventListener("beforeunload", function (ev) {
      if (keyOnScreen) { ev.preventDefault(); ev.returnValue = ""; }
    });

    var cb = callbackState(window.location.hash);
    if (cb && window.history && window.history.replaceState) window.history.replaceState(null, "", window.location.pathname);
    if (cb && cb.verified) {
      show("s-register");
      tell("15분 안에 패스키를 만들어야 합니다.");
    } else if (cb && cb.error) {
      show("s-start");
      fail({ kind: "api", status: 0, code: cb.error });
    } else {
      run(null, "", function () { return loadMe(true); });
    }
  }

  init();
})();
