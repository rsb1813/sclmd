/* ssalmuck-ai: the pages' shared script (design.md 6.23). GET /v1/status fills the limits table (once) and the live
   state (every 5 s while the tab is visible); buttons with data-copy copy a code block. No storage, no other origin,
   no inline code (the CSP forbids it). */
(function () {
  "use strict";

  // 6.20's numbers: shown when GET /v1/status cannot be read
  var FALLBACK = {
    limits: {
      ip_unverified: { tph: 300000, tpd: 7500000 },
      ip_verified: { tph: 750000, tpd: 15000000 },
      key: { tph: 1250000, tpd: 40000000 },
      output_weight: 5,
      cached_weight: 0.1,
      concurrency: { priority: 1, normal: 3, low: 8 }
    }
  };
  var TIERS = { ip_unverified: "미검증 IP", ip_verified: "검증 IP", key: "키", local: "운영자 PC" };
  var POLL_MS = 5000;      // the live state, while the tab is visible
  var TIMEOUT_MS = 10000;  // one GET /v1/status

  function tokens(n) {
    // 1,250,000 -> "125만", 40,000,000 -> "4,000만", 7,500,000 -> "750만"
    if (typeof n !== "number" || !isFinite(n) || n < 0) return "?";
    if (n >= 10000 && n % 10000 === 0) return (n / 10000).toLocaleString("ko-KR") + "만";
    return n.toLocaleString("ko-KR");
  }

  function duration(s) {
    if (typeof s !== "number" || !isFinite(s) || s < 0) return "?";
    s = Math.floor(s);
    var d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
    if (d) return d + "일 " + h + "시간";
    if (h) return h + "시간 " + m + "분";
    if (m) return m + "분";
    return s + "초";
  }

  // a live number: tokens a second with one decimal, counts whole; anything else (absent, null) a dash
  function live(v, decimals) {
    if (typeof v !== "number" || !isFinite(v) || v < 0) return "-";
    return decimals ? v.toLocaleString("ko-KR", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) : Math.round(v).toLocaleString("ko-KR");
  }

  function pick(obj, path) {
    var parts = path.split(".");
    for (var i = 0; i < parts.length; i++) {
      if (obj === null || typeof obj !== "object" || !(parts[i] in obj)) return undefined;
      obj = obj[parts[i]];
    }
    return obj;
  }

  function fillLimits(limits, fromServer) {
    var cells = document.querySelectorAll("[data-limit]");
    for (var i = 0; i < cells.length; i++) {
      var v = pick(limits, cells[i].getAttribute("data-limit"));
      if (typeof v === "number") {
        var kind = cells[i].getAttribute("data-format");
        cells[i].textContent = kind === "plain" ? v.toLocaleString("ko-KR") : tokens(v);
      }
    }
    var c = limits && limits.concurrency, refused = document.querySelectorAll("[data-limit-refused]");
    if (c && typeof c.priority === "number" && typeof c.normal === "number" && typeof c.low === "number") {
      for (var k = 0; k < refused.length; k++) refused[k].textContent = String(c.priority + c.normal + c.low + 1);
    }
    var src = document.querySelectorAll("[data-limits-source]");
    for (var j = 0; j < src.length; j++) {
      src[j].textContent = fromServer ? "서버가 알려 준 현재 값입니다." : "서버에 연결하지 못해 공개한 기본값을 표시합니다.";
    }
  }

  function fillCaller(tier) {
    var rows = document.querySelectorAll("tr[data-tier]");
    for (var i = 0; i < rows.length; i++) {
      rows[i].classList.toggle("you", rows[i].getAttribute("data-tier") === tier);
    }
    var els = document.querySelectorAll("[data-caller]");
    for (var j = 0; j < els.length; j++) {
      els[j].textContent = TIERS[tier] ? "이 브라우저의 등급은 " + TIERS[tier] + "입니다." : "";
      els[j].hidden = !TIERS[tier];
    }
  }

  // st: the status object, or null when it could not be read (the server is off, or no answer in 10 s)
  function stateText(f, st) {
    if (f === "ok") return !st ? "꺼짐" : st.ok === true ? "정상" : "점검 중";
    if (!st) return "-";
    var v = pick(st, f);
    if (f.indexOf("live.") === 0 || f.indexOf("queue.") === 0) return live(v, /tok_s/.test(f));
    if (f === "uptime_s") return duration(v);
    if (f.indexOf("zdr.") === 0) {
      if (f === "zdr.cache_ram_minutes" || f === "zdr.cache_disk_hours") {
        if (!v) return v === undefined ? "?" : "꺼짐";
        if (typeof v === "number") return v + (f === "zdr.cache_ram_minutes" ? "분" : "시간");
      }
      return v === false ? "아니요" : v === true ? "예" : v === undefined || v === "" ? "?" : v === null ? "없음" : String(v);
    }
    return v === undefined || v === null || v === "" ? "?" : String(v);
  }

  function fillState(st) {
    var fields = document.querySelectorAll("[data-state]");
    for (var j = 0; j < fields.length; j++) {
      var text = stateText(fields[j].getAttribute("data-state"), st), unit = fields[j].getAttribute("data-unit");
      if (unit && text !== "-") text += unit;  // a number that ends its line carries its unit: nothing after it can move
      if (fields[j].textContent !== text) fields[j].textContent = text;
    }
    var when = document.querySelectorAll("[data-state-time]");
    var now = new Date().toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
    for (var k = 0; k < when.length; k++) when[k].textContent = now + " 기준";
  }

  var polls = false, timer = null, inflight = false, limitsFromServer = false;

  function visible() { return document.visibilityState !== "hidden"; }

  function schedule() {
    if (timer) clearTimeout(timer);
    timer = polls && visible() ? setTimeout(loadStatus, POLL_MS) : null;
  }

  function loadStatus() {
    timer = null;
    if (inflight || typeof fetch !== "function") return;
    inflight = true;
    var ctl = typeof AbortController === "function" ? new AbortController() : null;
    var stop = ctl ? setTimeout(function () { ctl.abort(); }, TIMEOUT_MS) : null;
    fetch("/v1/status", { method: "GET", credentials: "omit", cache: "no-store", signal: ctl ? ctl.signal : undefined })
      .then(function (r) { if (!r.ok) throw new Error("status " + r.status); return r.json(); })
      .then(function (st) {
        if (!st || typeof st !== "object") throw new Error("not an object");
        var limits = st.limits && typeof st.limits === "object" ? st.limits : null;
        if (limits || !limitsFromServer) fillLimits(limits || FALLBACK.limits, !!limits);
        limitsFromServer = limitsFromServer || !!limits;
        fillCaller(st.caller ? st.caller.tier : null);
        fillState(st);
      })
      .catch(function () {
        if (!limitsFromServer) fillLimits(FALLBACK.limits, false);
        fillState(null);
      })
      .then(function () {
        if (stop) clearTimeout(stop);
        inflight = false;
        schedule();
      });
  }

  function startStatus() {
    if (!document.querySelector("[data-limit], [data-state], [data-caller]")) return;
    polls = !!document.querySelector("[data-state]");  // the limits alone: read once
    document.addEventListener("visibilitychange", function () {
      if (!polls) return;
      if (!visible()) { if (timer) clearTimeout(timer); timer = null; }
      else if (!inflight) loadStatus();
    });
    loadStatus();
  }

  function copyText(text, button) {
    function done(ok) {
      var old = button.getAttribute("data-label") || button.textContent;
      button.setAttribute("data-label", old);
      button.textContent = ok ? "복사했습니다" : "복사하지 못했습니다. 직접 선택해 복사하세요";
      setTimeout(function () { button.textContent = old; }, 2500);
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(fallbackCopy(text)); });
    } else {
      done(fallbackCopy(text));
    }
  }

  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.className = "offscreen";
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }

  function wireCopy() {
    var buttons = document.querySelectorAll("button[data-copy]");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].hidden = false;
      buttons[i].addEventListener("click", function (ev) {
        var target = document.getElementById(ev.currentTarget.getAttribute("data-copy"));
        if (target) copyText(target.textContent, ev.currentTarget);
      });
    }
  }

  window.SclmdSite = { tokens: tokens, duration: duration, live: live, copyText: copyText, FALLBACK: FALLBACK, TIERS: TIERS };

  if (typeof document !== "undefined" && document.addEventListener) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", function () { wireCopy(); startStatus(); });
    } else {
      wireCopy();
      startStatus();
    }
  }
})();
