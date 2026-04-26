(function() {
  // Force light mode — this UI is light-only; Gradio picks up system dark mode otherwise
  function enforceLightMode() {
    if (document.body && document.body.classList.contains('dark')) {
      document.body.classList.remove('dark');
    }
  }
  enforceLightMode();
  var _modeObs = new MutationObserver(enforceLightMode);
  _modeObs.observe(document.documentElement, { childList: true, subtree: false });
  document.addEventListener('DOMContentLoaded', function() {
    enforceLightMode();
    _modeObs.observe(document.body, { attributes: true, attributeFilter: ['class'] });
  });

  var source = null;
  var _runComplete = false;

  // phase ids → lane element ids
  var LANE_IDS = {
    model_a: 'lane-a',
    model_b: 'lane-b',
    model_c: 'lane-c',
    round:   'lane-round',
    judge:   'lane-judge'
  };
  var ORDER = ['model_a','model_b','model_c','round','judge'];

  function setLane(phase, state) {
    var id = LANE_IDS[phase]; if (!id) return;
    var el = document.getElementById(id); if (!el) return;
    el.setAttribute('data-state', state);
    var status = el.querySelector('.lane-status .label');
    if (status) {
      status.textContent = state === 'active' ? 'On the floor' :
                           state === 'done'   ? 'Yielded'      :
                                                'Awaiting';
    }
  }

  var currentPhase = null;

  function setStatus(label, mode) {
    var foot = document.getElementById('floor-foot');
    if (!foot) return;
    foot.setAttribute('data-conn', mode || 'idle');
    var lbl = foot.querySelector('.label');
    if (lbl) lbl.textContent = label;
  }

  function clearAll() {
    ORDER.forEach(function(p) {
      setLane(p, 'idle');
      var id = LANE_IDS[p];
      var el = document.getElementById(id);
      if (el) {
        var s = el.querySelector('.lane-stream');
        if (s) s.textContent = '';
      }
    });
    currentPhase = null;
  }

  function appendToken(token) {
    if (!currentPhase) currentPhase = 'model_a';
    var id = LANE_IDS[currentPhase]; if (!id) return;
    var el = document.getElementById(id); if (!el) return;
    var s = el.querySelector('.lane-stream');
    if (!s) return;
    s.textContent += token;
    s.scrollTop = s.scrollHeight;
  }

  function startPhase(phase) {
    if (phase === 'reset') {
      clearAll();
      return;
    }
    if (_runComplete) {
      clearAll();
      _runComplete = false;
    }
    if (currentPhase && currentPhase !== phase) {
      setLane(currentPhase, 'done');
    }
    currentPhase = phase;
    setLane(phase, 'active');
  }

  function connect() {
    if (source) return;
    source = new EventSource('http://localhost:7861/stream');
    setStatus('Connecting', 'waiting');

    source.onmessage = function(e) {
      try {
        var d = JSON.parse(e.data);
        if (d.token != null) {
          appendToken(d.token);
          setStatus('Live · streaming', 'live');
        }
      } catch(_) {}
    };
    source.addEventListener('phase', function(e) {
      try {
        var d = JSON.parse(e.data);
        if (d.phase) startPhase(d.phase);
      } catch(_) {}
    });
    source.addEventListener('done', function() {
      if (currentPhase) setLane(currentPhase, 'done');
      currentPhase = null;
      _runComplete = true;
      setStatus('Court adjourned', 'idle');
      source.close(); source = null;
      setTimeout(function() {
        // Re-arm for the next turn — keep deliberation content visible
        setStatus('Awaiting query', 'idle');
        connect();
      }, 1800);
    });
    source.onerror = function() {
      if (source) { source.close(); source = null; }
      setStatus('Reconnecting', 'waiting');
      setTimeout(connect, 1500);
    };
  }

  var obs = new MutationObserver(function() {
    if (document.getElementById('lane-a') && document.getElementById('floor-foot')) {
      obs.disconnect();
      setStatus('Awaiting query', 'idle');
      connect();
    }
  });
  obs.observe(document.body, { childList: true, subtree: true });
})();

;(function() {
  function watchClear() {
    var btn = document.getElementById('clear-btn');
    if (btn) {
      btn.addEventListener('click', function() {
        clearAll();
        _runComplete = false;
        setStatus('Awaiting query', 'idle');
      });
      return;
    }
    setTimeout(watchClear, 300);
  }
  watchClear();
})();
