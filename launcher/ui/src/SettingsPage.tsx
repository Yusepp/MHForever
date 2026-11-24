import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { motion } from 'motion/react';
import { api, regionTag, selectedGame, type SaveImportResult, type SaveScan, type Settings, type State, type UpdateInfo } from './bridge';
import { CopyIcon } from './icons';
import { FdxOptions } from './FdxOptions';
import { FucOptions } from './FucOptions';
import { OnlineSettings } from './OnlineSettings';
import { TexturePackSection } from './TexturePack';
import { anisotropyLevels, frameRates, hudSizes, resolutions, songPlaces } from './format';
import { setNavigationPaused } from './gamepad';
import { keyGroups, keyLabel, reservedKeys } from './keys';
import { anySourcePressed, assignSource, padGroups, padLabel, pressedSource, stickChoices } from './pad';

const rise = (delay: number) => ({
  initial: { opacity: 0, y: 18, filter: 'blur(6px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)' },
  transition: { duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] as const },
});

// The languages every regional disc of the game offers. '' leaves the choice to the game's own language screen, which is
// what the original shows at every start; picking one makes the game boot in that language and answer its screen
// (docs/06-mhfu-module.md).
const languages = [['', 'Ask'], ['en', 'EN'], ['fr', 'FR'], ['de', 'DE'], ['es', 'ES'], ['it', 'IT']] as const;
const languageNames: Record<string, string> = { en: 'English', fr: 'Français', de: 'Deutsch', es: 'Español', it: 'Italiano' };

type Props = {
  state: State;
  onChange: (s: Partial<Settings>) => void;
  onPseudonym: (name: string) => void;
  onCopyCode: () => void;
  onChangeCopy: () => void;
  onRemoveCopy: (id: string) => void;
  onLocateExecutable: () => void;
};

function Toggle({ on, onToggle, label }: { on: boolean; onToggle: () => void; label: string }) {
  return <button className={'tog' + (on ? ' on' : '')} onClick={onToggle} role="switch" aria-checked={on} aria-label={label} />;
}

function ToggleRow({ label, help, on, onToggle, prose }: {
  label: string; help: string; on: boolean; onToggle: () => void; prose?: boolean;
}) {
  return (
    <div className="srow">
      <div className="slabel"><b>{label}</b><span className={prose ? 'prose' : undefined}>{help}</span></div>
      <Toggle on={on} onToggle={onToggle} label={label} />
    </div>
  );
}

function Segmented<T extends string | number>({ label, value, options, onSelect, hidePill }: {
  label: string;
  value: T;
  options: readonly { value: T; label: string }[];
  onSelect: (value: T) => void;
  hidePill?: boolean;
}) {
  const index = Math.max(0, options.findIndex((o) => o.value === value));
  return (
    <div className="seg wide" role="radiogroup" aria-label={label} style={{ '--n': options.length } as CSSProperties}>
      {!hidePill && <span className="pill" style={{ transform: `translateX(${index * 100}%)` }} />}
      {options.map((o) => (
        <button key={String(o.value)} className={value === o.value ? 'on' : ''} role="radio" aria-checked={value === o.value}
          onClick={() => onSelect(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

function Volume({ id, label, value, onCommit }: { id: string; label: string; value: number; onCommit: (v: number) => void }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return (
    <div className="srow">
      <label htmlFor={id} className="slabel" style={{ flex: '0 0 160px' }}><b>{label}</b></label>
      <div className="slider">
        <input
          id={id}
          className="rng"
          type="range"
          min={0}
          max={100}
          value={v}
          style={{ ['--v' as string]: `${v}%` }}
          onChange={(e) => { const n = Number(e.target.value); setV(n); onCommit(n); }}
        />
        <output htmlFor={id}>{v}</output>
      </div>
    </div>
  );
}

// A setting in percent between `min` and `max`, in steps of `step`, with its help line.
function PercentSlider({ id, label, help, min, max, step, value, onCommit }: {
  id: string; label: string; help: string; min: number; max: number; step: number; value: number; onCommit: (v: number) => void;
}) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return (
    <div className="srow">
      <label htmlFor={id} className="slabel"><b>{label}</b><span className="prose">{help}</span></label>
      <div className="slider" style={{ flex: '0 1 300px' }}>
        <input
          id={id}
          className="rng"
          type="range"
          min={min}
          max={max}
          step={step}
          value={v}
          style={{ ['--v' as string]: `${((v - min) / (max - min)) * 100}%` }}
          onChange={(e) => { const n = Number(e.target.value); setV(n); onCommit(n); }}
        />
        <output htmlFor={id} style={{ width: 52 }}>{v}%</output>
      </div>
    </div>
  );
}

// Bringing in saves made by a PSP or PPSSPP: choose the folder, see what was found, confirm.
// Where a copy of the game keeps its saves: its own folder inside the saves folder (launcher install::game_saves_dir).
function copySavesPath(folder: string, discId: string): string {
  const sep = folder.includes('\\') ? '\\' : '/';
  return [folder.replace(/[\\/]+$/, ''), discId, 'PSP', 'SAVEDATA'].join(sep);
}

function ImportSaves({ gameId }: { gameId: string | undefined }) {
  const [scan, setScan] = useState<SaveScan | null>(null);
  const [done, setDone] = useState<SaveImportResult | null>(null);
  const [busy, setBusy] = useState(false);
  const choose = async () => {
    if (!gameId || busy) return;
    setBusy(true);
    setDone(null);
    try {
      const r = await api.mhf_scanSaves(gameId);
      if (r.picked || r.error) setScan(r);
    } finally {
      setBusy(false);
    }
  };
  const apply = async () => {
    setBusy(true);
    try {
      setDone(await api.mhf_importSaves());
      setScan(null);
    } finally {
      setBusy(false);
    }
  };
  const found = scan?.found ?? [];
  const replaces = found.some((f) => f.exists);
  return (
    <>
      <div className="srow">
        <div className="slabel">
          <b>Import saves</b>
          <span>Imports saves from a PSP memory stick or from PPSSPP. Select the SAVEDATA folder or the memory stick folder. Only saves from the same version of the game are supported.</span>
        </div>
        <button className="ghost mid" onClick={choose} disabled={!gameId || busy}>Choose folder…</button>
      </div>
      {scan && (
        <div className="srow" style={{ flexDirection: 'column', alignItems: 'stretch', gap: 10 }}>
          {scan.error && <span className="muted">{scan.error}</span>}
          {found.length === 0 && !scan.error && <span className="muted">{scan.note}</span>}
          {found.map((f) => (
            <div key={f.folder} className="slabel">
              <b>{f.title || 'Game data'} <span style={{ opacity: 0.6 }}>({f.folder})</span></b>
              <span>{f.exists ? 'Replaces your current save. A backup is kept.' : 'New save.'}</span>
            </div>
          ))}
          {found.length > 0 && (
            <div className="row-wrap" style={{ gap: 10 }}>
              <button className={replaces ? 'ghost mid danger' : 'ghost mid'} onClick={apply} disabled={busy}>
                {found.length === 1 ? 'Import this save' : `Import ${found.length} saves`}
              </button>
              <button className="ghost mid" onClick={() => setScan(null)} disabled={busy}>Cancel</button>
            </div>
          )}
        </div>
      )}
      {done && (
        <div className="srow">
          <div className="slabel">
            <b>{done.error ? 'The import did not finish' : done.imported === 1 ? 'Save imported' : `${done.imported} saves imported`}</b>
            <span>
              {done.error || 'Load it from the game after pressing Play.'}
              {done.backup && ` Previous save backed up to ${done.backup}.`}
            </span>
          </div>
        </div>
      )}
    </>
  );
}

// Keyboard bindings: click a key and press the one you want. Two buttons can never share a key (the other one takes
// the key this one had), Backspace leaves a button without a key, Esc cancels (docs/07-launcher-ux.md section 5).
function KeyboardControls({ keys, defaults, onChange }: {
  keys: Record<string, string>;
  defaults: Record<string, string>;
  onChange: (s: Partial<Settings>) => void;
}) {
  const [capturing, setCapturing] = useState<string | null>(null);
  const [notice, setNotice] = useState('');
  const latest = useRef({ keys, onChange });
  latest.current = { keys, onChange };

  useEffect(() => {
    if (!capturing) return;
    setNavigationPaused(true); // the keys being recorded must not move the launcher around
    const record = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopImmediatePropagation();
      if (e.repeat) return;
      const action = capturing;
      if (e.code === 'Escape') {
        setCapturing(null);
        return;
      }
      if (reservedKeys[e.code]) {
        setNotice(`${keyLabel(e.code)} is kept by the game: ${reservedKeys[e.code]} Press another key, or Esc to cancel.`);
        return;
      }
      assign(action, e.code);
    };
    // A mouse button binds like a key (Mouse1 left, Mouse2 right, Mouse3 middle, Mouse4 and Mouse5 the side buttons).
    const mouseCodes = ['Mouse1', 'Mouse3', 'Mouse2', 'Mouse4', 'Mouse5']; // by MouseEvent.button
    const click = (e: MouseEvent) => {
      const code = mouseCodes[e.button];
      if (!code) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      assign(capturing as string, code);
    };
    const swallow = (e: MouseEvent) => { e.preventDefault(); e.stopImmediatePropagation(); }; // no menu, no "back"
    const assign = (action: string, code: string) => {
      setNotice('');
      const next = { ...latest.current.keys };
      const previous = next[action] ?? '';
      if (code !== 'Backspace' && code !== 'Delete') {
        for (const [other, value] of Object.entries(next)) {
          if (other !== action && value === code) next[other] = previous;
        }
        next[action] = code;
      } else {
        next[action] = '';
      }
      latest.current.onChange({ keys: next });
      setCapturing(null);
    };
    window.addEventListener('keydown', record, true);
    // After the click that started listening, so that click does not bind itself.
    const listen = window.setTimeout(() => {
      window.addEventListener('mousedown', click, true);
      window.addEventListener('mouseup', swallow, true);
      window.addEventListener('auxclick', swallow, true);
      window.addEventListener('contextmenu', swallow, true);
    }, 0);
    return () => {
      window.clearTimeout(listen);
      window.removeEventListener('mousedown', click, true);
      window.removeEventListener('mouseup', swallow, true);
      window.removeEventListener('auxclick', swallow, true);
      window.removeEventListener('contextmenu', swallow, true);
      window.removeEventListener('keydown', record, true);
      setNavigationPaused(false);
    };
  }, [capturing]);

  return (
    <>
      <div className="srow">
        <div className="slabel">
          <b>Keyboard</b>
          <span>Select an action, then press a key. Backspace clears it; Esc cancels.</span>
        </div>
        <button className="ghost mid" onClick={() => { setNotice(''); onChange({ keys: { ...defaults } }); }}>Restore defaults</button>
      </div>
      {notice && (
        <div className="srow">
          <span className="slabel" style={{ color: 'var(--warn)', fontSize: 13 }}>{notice}</span>
        </div>
      )}
      {keyGroups.map((group) => (
        <div className="srow" key={group.title}>
          <div className="slabel"><b>{group.title}</b></div>
          <div className="row-wrap" style={{ gap: 8 }}>
            {group.actions.map((action) => {
              const code = keys[action.id] ?? '';
              const listening = capturing === action.id;
              return (
                <button
                  key={action.id}
                  className={'keycap' + (listening ? ' listening' : '') + (code ? '' : ' unset')}
                  aria-label={`${action.label}: ${listening ? 'press a key' : keyLabel(code)}`}
                  onClick={() => { setNotice(''); setCapturing(listening ? null : action.id); }}
                >
                  <span className="keycap-name">{action.label}</span>
                  <span className="keycap-key">{listening ? 'Press a key or click…' : keyLabel(code)}</span>
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </>
  );
}

// Controller bindings: click a button and press the one you want on the controller. Two buttons never share a source (the
// other one takes the source this one had), Backspace or Delete on the keyboard leaves a button empty, Esc cancels. The
// controller's own navigation is paused while recording (docs/07-launcher-ux.md section 5).
function PadControls({ pad, defaults, onChange }: {
  pad: Record<string, string>;
  defaults: Record<string, string>;
  onChange: (s: Partial<Settings>) => void;
}) {
  const [capturing, setCapturing] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const latest = useRef({ pad, onChange });
  latest.current = { pad, onChange };

  useEffect(() => {
    const check = () => setConnected(Array.from(navigator.getGamepads ? navigator.getGamepads() : []).some((g) => g && g.connected));
    check();
    window.addEventListener('gamepadconnected', check);
    window.addEventListener('gamepaddisconnected', check);
    return () => {
      window.removeEventListener('gamepadconnected', check);
      window.removeEventListener('gamepaddisconnected', check);
    };
  }, []);

  useEffect(() => {
    if (!capturing) return;
    setNavigationPaused(true);
    let frame = 0;
    let released = !anySourcePressed(); // the press that started the recording must not be the answer
    const poll = () => {
      const pads = navigator.getGamepads ? navigator.getGamepads() : [];
      let pressed: string | null = null;
      for (const g of pads) {
        if (g && g.connected && pressed === null) pressed = pressedSource(g);
      }
      if (pressed === null) released = true;
      else if (released) {
        latest.current.onChange({ pad: assignSource(latest.current.pad, capturing, pressed) });
        setCapturing(null);
        return;
      }
      frame = requestAnimationFrame(poll);
    };
    frame = requestAnimationFrame(poll);
    const key = (e: KeyboardEvent) => {
      if (e.code === 'Escape') {
        e.preventDefault();
        e.stopImmediatePropagation();
        setCapturing(null);
      } else if (e.code === 'Backspace' || e.code === 'Delete') {
        e.preventDefault();
        e.stopImmediatePropagation();
        latest.current.onChange({ pad: { ...latest.current.pad, [capturing]: '' } });
        setCapturing(null);
      }
    };
    window.addEventListener('keydown', key, true);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('keydown', key, true);
      setNavigationPaused(false);
    };
  }, [capturing]);

  const stick = pad.stick ?? 'left';
  return (
    <>
      <div className="srow">
        <div className="slabel">
          <b>Controller</b>
          <span>
            {connected
              ? 'Select an action, then press a controller button. Backspace clears it; Esc cancels.'
              : 'No controller detected. Connect a controller and press any button.'}
          </span>
        </div>
        <button className="ghost mid" onClick={() => onChange({ pad: { ...defaults } })}>Restore defaults</button>
      </div>
      {padGroups.map((group) => (
        <div className="srow" key={group.title}>
          <div className="slabel"><b>{group.title}</b></div>
          <div className="row-wrap" style={{ gap: 8 }}>
            {group.actions.map((action) => {
              const value = pad[action.id] ?? '';
              const listening = capturing === action.id;
              return (
                <button
                  key={action.id}
                  className={'keycap' + (listening ? ' listening' : '') + (value ? '' : ' unset')}
                  aria-label={`${action.label}: ${listening ? 'press a button' : padLabel(value)}`}
                  onClick={() => setCapturing(listening ? null : action.id)}
                >
                  <span className="keycap-name">{action.label}</span>
                  <span className="keycap-key">{listening ? 'Press a button…' : padLabel(value)}</span>
                </button>
              );
            })}
          </div>
        </div>
      ))}
      <div className="srow">
        <div className="slabel"><b>Analog stick</b><span>Stick used to move the character.</span></div>
        <Segmented label="Analog stick" value={stick} options={stickChoices} onSelect={(v) => onChange({ pad: { ...pad, stick: v } })} />
      </div>
    </>
  );
}

function Controls({ keys, pad, keyDefaults, padDefaults, onChange }: {
  keys: Record<string, string>;
  pad: Record<string, string>;
  keyDefaults: Record<string, string>;
  padDefaults: Record<string, string>;
  onChange: (s: Partial<Settings>) => void;
}) {
  const [device, setDevice] = useState<'keyboard' | 'controller'>('keyboard');
  const options = [{ value: 'keyboard', label: 'Keyboard' }, { value: 'controller', label: 'Controller' }] as const;
  return (
    <>
      <div className="srow">
        <div className="slabel"><b>Edit the controls of</b><span>Keyboard and controller can be used at the same time.</span></div>
        <Segmented label="Device" value={device} options={options} onSelect={setDevice} />
      </div>
      {device === 'keyboard'
        ? <KeyboardControls keys={keys} defaults={keyDefaults} onChange={onChange} />
        : <PadControls pad={pad} defaults={padDefaults} onChange={onChange} />}
    </>
  );
}

export function SettingsPage({ state, onChange, onPseudonym, onCopyCode, friendCode, onSavesFolder, update, onInstallUpdate, say }: Pick<Props, 'state' | 'onChange' | 'onPseudonym' | 'onCopyCode'> & {
  friendCode: string;
  onSavesFolder: (backToDefault: boolean) => void;
  update: UpdateInfo | null;
  onInstallUpdate: () => void;
  say: (text: string) => void;
}) {
  const s = state.settings;
  const [name, setName] = useState(state.pseudonym);
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [updateNote, setUpdateNote] = useState<string | null>(null);
  useEffect(() => setName(state.pseudonym), [state.pseudonym]);
  const checkNow = () => {
    if (checkingUpdate) return;
    setCheckingUpdate(true);
    setUpdateNote(null);
    api.mhf_checkUpdate().then((res) => {
      setCheckingUpdate(false);
      if (res.status === 'available') {
        setUpdateNote(`Version ${res.update.version} is available.`);
      } else if (res.status === 'upToDate') {
        setUpdateNote(`MHForever ${res.current} is up to date.`);
        say('MHForever is up to date');
      } else {
        setUpdateNote(res.message);
        say('Update check failed');
      }
    });
  };
  const commitName = () => {
    const clean = name.trim().slice(0, 16);
    if (clean && clean !== state.pseudonym) onPseudonym(clean);
    else setName(state.pseudonym);
  };
  return (
    <section className="settings">
      <motion.div className="page-title" {...rise(0)}>
        <h2>Settings</h2>
        <p>These apply to MHForever and to all your games. Each game has its own video, audio and controls under Game settings.</p>
      </motion.div>

      <motion.div className="glass section" {...rise(0.18)}>
        <h3>Profile</h3>
        <div className="srow">
          <label htmlFor="pseudonym" className="slabel"><b>Name</b><span>Shown to friends and entered when the game asks for a name.</span></label>
          <input
            id="pseudonym"
            className="text-input"
            value={name}
            maxLength={16}
            onChange={(e) => setName(e.target.value)}
            onBlur={commitName}
            onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
            style={{ width: 240 }}
          />
        </div>
        <div className="srow">
          <div className="slabel"><b>Friend code</b><span>Stored on this computer only. Back it up from the profile menu.</span></div>
          <div className="code-box" style={{ minWidth: 300 }}>
            <span className="code">{friendCode}</span>
            <button className="copy" onClick={onCopyCode} aria-label="Copy friend code"><CopyIcon /> Copy</button>
          </div>
        </div>
      </motion.div>
      <motion.div className="glass section" {...rise(0.2)}>
        <h3>Online (experimental)</h3>
        <OnlineSettings server={s.adhocServer ?? ''} onChange={(adhocServer) => onChange({ adhocServer })} />
      </motion.div>

      <motion.div className="glass section" {...rise(0.1)}>
        <h3>Storage</h3>
        <ToggleRow
          label="Keep a copy of my disc image"
          help="Stores a copy of the disc image in MHForever's folder when a game is added. Requires about 1 GB."
          on={s.copyImage}
          onToggle={() => onChange({ copyImage: !s.copyImage })}
        />
        <div className="srow">
          <div className="slabel">
            <b>Saves folder</b>
            <span className="prose">
              Location of your saves. Each copy of the game saves in its own folder inside it (for example
              ULUS10391\PSP\SAVEDATA), shown in its Game settings. Changing it copies your saves to the new folder;
              existing saves there are kept.
            </span>
            <span className="path">{state.savesFolder}</span>
          </div>
          <div className="row-actions">
            <button className="ghost mid" onClick={() => onSavesFolder(false)}>Change…</button>
            {state.savesFolderCustom && <button className="ghost mid" onClick={() => onSavesFolder(true)}>Use default</button>}
          </div>
        </div>
      </motion.div>
      <motion.div className="glass section" {...rise(0.3)}>
        <h3>About</h3>
        <div className="srow">
          <div className="slabel">
            <b>MHForever {state.version}</b>
            <span>{update ? `Version ${update.version} is ready to install. Your saves, paths and settings are kept.` : (updateNote ?? 'Free software under GPL-3.0 or later. Contains no game data and requires your own copy of the game.')}</span>
          </div>
          <div className="row-actions">
            {update ? (
              <button className="ghost mid" onClick={onInstallUpdate}>Update to {update.version}</button>
            ) : (
              <button className="ghost mid" onClick={checkNow} disabled={checkingUpdate}>{checkingUpdate ? 'Checking…' : 'Check now'}</button>
            )}
          </div>
        </div>
        <ToggleRow
          label="Check for updates"
          help="Checks GitHub for a new release at startup. No other data is sent."
          prose
          on={s.checkUpdates ?? true}
          onToggle={() => onChange({ checkUpdates: !(s.checkUpdates ?? true) })}
        />
      </motion.div>
    </section>
  );
}

export function GameSettingsPage({ state, onChange, onChangeCopy, onRemoveCopy, onLocateExecutable, onState, say }: Omit<Props, 'onPseudonym' | 'onCopyCode'> & {
  onState: (s: State) => void;
  say: (text: string) => void;
}) {
  const s = state.settings;
  const game = selectedGame(state) ?? state.games[0];
  const fucomplete = game?.discId === 'ULJM05500-FUC'; // FUComplete's own options apply
  // Freedom Unite's own enhancements (widescreen, HUD scale, the Game section); other games have none yet. Frame
  // interpolation is the frontend's and works for every game.
  const enhanced = !game || game.id === 'mhfu';
  return (
    <section className="settings">
      <motion.div className="page-title" {...rise(0)}>
        <h2>{game ? `${game.title} (${regionTag(game)})` : 'Game settings'}</h2>
        <p>Saved automatically for this copy of the game. Changes apply the next time the game starts.</p>
      </motion.div>

      <motion.div className="glass section" {...rise(0.06)}>
        <h3>Video</h3>
        <div className="srow">
          <div className="slabel">
            <b>Graphics backend</b>
            <span>{(s.graphicsBackend ?? 'opengl') === 'vulkan' ? 'Vulkan 1.3+ (experimental).' : (s.graphicsBackend ?? 'opengl') === 'software' ? 'CPU software renderer.' : 'OpenGL 3.3 (default).'}</span>
          </div>
          <Segmented
            label="Graphics backend"
            value={s.graphicsBackend ?? 'opengl'}
            options={[{ value: 'opengl', label: 'OpenGL' }, { value: 'vulkan', label: 'Vulkan' }, { value: 'software', label: 'Software' }]}
            onSelect={(graphicsBackend) => onChange({ graphicsBackend })}
          />
        </div>
        <div className="srow">
          <div className="slabel"><b>Internal resolution</b><span>{resolutions[s.scale - 1]}{s.scale === 4 ? ' (recommended)' : ''}</span></div>
          <Segmented
            label="Internal resolution"
            value={s.scale}
            options={resolutions.map((_, i) => ({ value: i + 1, label: `${i + 1}×` }))}
            onSelect={(scale) => onChange({ scale })}
          />
        </div>
        <div className="srow">
          <div className="slabel"><b>Frame rate</b><span>{s.fps === 30 ? 'Original frame rate.' : `${s.fps} fps with interpolated frames. Experimental.`}</span></div>
          <Segmented
            label="Frame rate"
            value={s.fps}
            options={frameRates.map((n) => ({ value: n, label: String(n) }))}
            onSelect={(fps) => onChange({ fps })}
          />
        </div>
        <ToggleRow label="Fullscreen" help="Starts the game in fullscreen. Press F11 or Alt+Enter to switch while playing."
          on={s.fullscreen} onToggle={() => onChange({ fullscreen: !s.fullscreen })} />
        {enhanced && <ToggleRow label="Widescreen" help="Fills wide windows and screens with a wider view."
          on={!!s.widescreen} onToggle={() => onChange({ widescreen: !s.widescreen })} />}
        <ToggleRow label="Smooth texture filtering" help="Applies bilinear filtering to the game's textures."
          on={s.smoothTextures} onToggle={() => onChange({ smoothTextures: !s.smoothTextures })} />
        <div className="srow">
          <div className="slabel">
            <b>Text font</b>
            <span>{s.textFontName ? `${s.textFontName}. Large fonts may overlap.` : 'Default font with HD text. Choose a .ttf or .otf file to use your own.'}</span>
          </div>
          <div className="row-wrap" style={{ gap: 10 }}>
            <button className="ghost mid" onClick={() => api.mhf_textFont('choose').then((r) => {
              onState(r.state);
              if (r.error) say(r.error);
              else if (r.picked) say('Font set');
            })}>Choose…</button>
            {s.textFontName && <button className="ghost mid" onClick={() => api.mhf_textFont('default').then((r) => onState(r.state))}>Default</button>}
          </div>
        </div>
        {enhanced && <div className="srow">
          <div className="slabel">
            <b>HUD scale</b>
            <span>{(s.hudSize ?? 100) === 100 ? 'Original size.' : `Quest HUD at ${s.hudSize}% of its original size.`}</span>
          </div>
          <Segmented
            label="HUD scale"
            value={s.hudSize ?? 100}
            options={hudSizes.map((n) => ({ value: n, label: `${n}%` }))}
            onSelect={(hudSize) => onChange({ hudSize })}
          />
        </div>}
      </motion.div>
      {(s.graphicsBackend ?? 'opengl') === 'vulkan' && <motion.div className="glass section" {...rise(0.062)}>
        <h3>Vulkan Rendering</h3>
        <ToggleRow
          label="Enhanced lighting"
          help="Enables modern Vulkan lighting, shadows, and visual effects. Can also be adjusted live in-game from the Esc menu."
          on={!!s.enhancedLighting}
          onToggle={() => onChange({ enhancedLighting: !s.enhancedLighting })}
        />
        {s.enhancedLighting && (
          <>
            {([
              ['vkAo', 'Ambient occlusion', 'Multi-bounce horizon shading inside crevices and corners.'],
              ['vkContactShadows', 'Contact & sun shadows', 'Short-range contact grounding and long projected sun shadows.'],
              ['vkSsgi', 'Indirect bounce & flash', 'Colored firelight/lava bounce and dynamic explosion flashes.'],
              ['vkSpecular', 'Specular & sky bounce', 'Biome specular sheen, Fresnel rim light, and hemisphere bounce.'],
              ['vkVolumetricFog', 'Volumetric fog & shafts', 'Biome height fog, outdoor sun shafts, and cave doorway rays.'],
              ['vkBloom', 'Emissive bloom & shimmer', 'Selective glow on fire, lava, and lightning with rising heat haze.'],
              ['vkSoftParticles', 'Soft 3D particles', 'Depth-softened smoke, snow, fire, and monster breath.'],
              ['vkFxaa', 'Pre-HUD anti-aliasing', 'Smooths 3D edges before the 2D HUD is drawn.'],
              ['vkVignette', 'Moody vignette', 'Subtle edge framing in caves, swamps, volcanoes, and sieges.'],
              ['vkKillcamDof', 'Killcam depth of field', 'Peripheral bokeh blur during the monster killcam.'],
            ] as const).map(([key, label, help]) => (
              <ToggleRow key={key} label={label} help={help} on={s[key] ?? true} onToggle={() => onChange({ [key]: !(s[key] ?? true) })} />
            ))}
          </>
        )}
      </motion.div>}

      {enhanced && <motion.div className="glass section" {...rise(0.065)}>
        <h3>Gameplay QoL</h3>
        <ToggleRow
          label="Show true weapon values"
          help={fucomplete
            ? "Uses FUComplete's true raw display for attack, element and status."
            : "Displays attack and element without the game's display multipliers."}
          on={s.trueWeaponValues}
          onToggle={() => onChange({ trueWeaponValues: !s.trueWeaponValues })}
        />
        <ToggleRow label="Fast supply box" help="Shortens the delay between items taken from the supply box."
          on={s.fastSupplyBox ?? false} onToggle={() => onChange({ fastSupplyBox: !(s.fastSupplyBox ?? false) })} />
        <ToggleRow label="Dynamic Guild quest scaling" help="Scales large-monster HP and part stagger in Guild quests to the active hunter count (1–4 players)."
          on={s.guildQuestScaling ?? false} onToggle={() => onChange({ guildQuestScaling: !(s.guildQuestScaling ?? false) })} />
        <ToggleRow label="Quest auto-clear" help="Hold L + R + Select for 2 seconds during a quest to complete it."
          on={s.questAutoclear ?? false} onToggle={() => onChange({ questAutoclear: !(s.questAutoclear ?? false) })} />
        <ToggleRow label="Smithy: browse with panels open" help="Keeps the smithy list scrollable while the materials or Info panel is open."
          on={s.smithyBrowse ?? false} onToggle={() => onChange({ smithyBrowse: !(s.smithyBrowse ?? false) })} />
        {s.smithyBrowse && (
          <ToggleRow label="Smithy: improved layout" help="Shows the Info panel beside the item list and its materials."
            on={s.smithyLayout ?? false} onToggle={() => onChange({ smithyLayout: !(s.smithyLayout ?? false) })} />
        )}
        <ToggleRow label="Analog camera" help="The right stick turns and tilts the camera in proportion to how far it is pushed."
          on={s.analogCamera ?? false} onToggle={() => onChange({ analogCamera: !(s.analogCamera ?? false) })} />
        {s.analogCamera && (
          <>
            <PercentSlider id="camera-turn-speed" label="Analog camera turn speed" help="Turn speed at full stick deflection. 100% matches the D-pad." min={50} max={200} step={10} value={s.analogCameraTurnSpeed ?? 100} onCommit={(v) => onChange({ analogCameraTurnSpeed: v })} />
            <PercentSlider id="camera-tilt-speed" label="Analog camera tilt speed" help="Tilt speed at full stick deflection." min={50} max={200} step={10} value={s.analogCameraTiltSpeed ?? 100} onCommit={(v) => onChange({ analogCameraTiltSpeed: v })} />
            <ToggleRow label="Mouse camera" help="The mouse controls the camera and bow and bowgun aiming. Press Esc to release the mouse."
              on={s.mouseCamera ?? false} onToggle={() => onChange({ mouseCamera: !(s.mouseCamera ?? false) })} />
            {s.mouseCamera && (
              <PercentSlider id="mouse-camera-speed" label="Mouse sensitivity" help="Camera sensitivity to mouse movement." min={25} max={300} step={25} value={s.mouseCameraSpeed ?? 100} onCommit={(v) => onChange({ mouseCameraSpeed: v })} />
            )}
          </>
        )}
        {([
          ['gatheringAssistance', 'Gathering assistance', 'Shows nearby gathering, fishing, and carving targets; Circle selects a usable pickaxe, bug net, or bait.'],
          ['infiniteGatheringTools', 'Infinite gathering tools', 'Mine, catch bugs, and fish without carrying pickaxes, bug nets, or bait (uses a temporary high-tier tool and removes it afterward).'],
          ['questBoxIcons', 'Quest camp box icons', 'Shows contextual gathering-style icons near the camp Blue Supply Box and Red Delivery Box (red X when carrying nothing to deliver).'],
          ['modernCarving', 'Modern carving', 'Hold Circle to repeat gathering, carving, mining, and bug-catching at the same spot.'],
          ['invulnerableCarveQuest', 'Invulnerable post-quest carving', 'Ignores hits, roars and wind while carving after Quest Clear; normal vulnerability resumes right away.'],
          ['keyQuestMarkers', 'Show key quest markers', 'Marks Village and Guild Hall key quests with a gold key, including completed quests.'],
          ['betterSharpness', 'Better sharpness gauge', 'Shows how much of each sharpness tier is left during a quest.'],
          ['damageNumbers', 'Floating damage numbers', 'Displays World/Rise/Wilds-style floating damage numbers when hitting monsters.'],
          ['capturableIndicator', 'Capturable monster indicator', "Turns a capturable large monster's minimap dot cyan instead of pink."],
          ['questStartTracker', 'Quest-start monster tracker (60s)', 'Reveals large monsters on the map for the first minute of each quest, then expires naturally.'],
          ['skipQuestClearTimer', 'Skip post-quest countdown', 'Hold Select after Quest Clear to fast-forward the 60-second return countdown.'],
          ['infiniteWhetstone', 'Infinite whetstone', 'Sharpen without consuming Whetstones (supplies a temporary Whetstone if none carried).'],
          ['infinitePaintballs', 'Infinite paintballs', 'Throw Paintballs without consuming them (supplies a temporary Paintball if none carried).'],
          ['comboBooksFromBox', 'Combo books from item box', 'Guarantees 100% combine rate once you own all 5 Books of Combos (in Item Box or Pouch).'],
          ['itemPouchLoadouts', 'Hunter loadouts (Item & Armor sets)', 'Press Select in your House or Guild Hall Item Box to save and load up to 5 Item & Armor Sets.'],
          ['hornSongs', 'Show hunting horn songs', "Displays the hunting horn's songs, the song being played and active effects during quests."],
        ] as const).map(([key, label, help]) => (
          <ToggleRow key={key} label={label} help={help} on={s[key] ?? false} onToggle={() => onChange({ [key]: !(s[key] ?? false) })} />
        ))}
        {s.hornSongs && (
          <div className="srow">
            <div className="slabel"><b>Hunting horn songs position</b><span>{(s.hornSongsPlace ?? '').startsWith('custom:') ? 'Custom position set in game.' : 'Position of the song list. It can also be moved in game from Esc > QoL.'}</span></div>
            <Segmented
              label="Hunting horn songs position"
              value={s.hornSongsPlace ?? 'next-to-sheet'}
              options={songPlaces.map((p) => ({ value: p.id, label: p.label }))}
              onSelect={(hornSongsPlace) => onChange({ hornSongsPlace })}
              hidePill={(s.hornSongsPlace ?? '').startsWith('custom:')}
            />
          </div>
        )}
      </motion.div>}
      <motion.div className="glass section" {...rise(0.07)}>
        <h3>Texture pack</h3>
        <TexturePackSection state={state} onState={onState} say={say} toggle={(p) => <Toggle {...p} />} />
        {s.texturePack && s.texturePackEnabled && (
          <>
            <ToggleRow label="Mipmaps" help="Smooths distant textures and reduces shimmering."
              on={s.packMipmaps ?? false} onToggle={() => onChange({ packMipmaps: !(s.packMipmaps ?? false) })} />
            <div className="srow">
              <div className="slabel"><b>Anisotropic filtering</b><span>Sharpens textures viewed at an angle.</span></div>
              <Segmented
                label="Anisotropic filtering"
                value={s.anisotropy ?? 16}
                options={anisotropyLevels.map((n) => ({ value: n, label: n === 1 ? 'Off' : `${n}×` }))}
                onSelect={(anisotropy) => onChange({ anisotropy })}
              />
            </div>
          </>
        )}
      </motion.div>
      {game?.discId.includes('-FDX') && <motion.div className="glass section" {...rise(0.07)}>
        <h3>Freedom DX</h3>
        <div className="srow">
          <div className="slabel">
            <b>FDXDAT folder</b>
            <span>{state.fdxData
              ? 'In place: its event quests are available.'
              : 'Needed for the event quests. Choose the FDXDAT folder from your Freedom DX download.'}</span>
          </div>
          <button className="ghost mid" onClick={() => api.mhf_fdxData().then((r) => {
            onState(r.state);
            if (r.error) say(r.error);
            else if (r.picked) say('FDXDAT folder copied');
          })}>{state.fdxData ? 'Replace…' : 'Choose…'}</button>
        </div>
        <FdxOptions s={s} onChange={onChange} toggle={(p) => <Toggle {...p} />} />
      </motion.div>}
      {fucomplete && <motion.div className="glass section" {...rise(0.07)}>
        <h3>FUComplete</h3>
        <FucOptions s={s} onChange={onChange} toggle={(p) => <Toggle {...p} />} />
      </motion.div>}
      {game?.market !== 'JP' && <motion.div className="glass section" {...rise(0.07)}>
        <h3>Language</h3>
        <div className="srow">
          <div className="slabel">
            <b>Language</b>
            <span>
              {s.language === ''
                ? 'The game asks for a language at every start.'
                : `The game starts in ${languageNames[s.language] ?? s.language}.`}
            </span>
          </div>
          <Segmented
            label="Language"
            value={s.language}
            options={languages.map(([id, label]) => ({ value: id, label }))}
            onSelect={(language) => onChange({ language })}
          />
        </div>
      </motion.div>}
      <motion.div className="glass section" {...rise(0.08)}>
        <h3>Controls</h3>
        <Controls keys={s.keys ?? {}} pad={s.pad ?? {}} keyDefaults={state.keyDefaults ?? {}} padDefaults={state.padDefaults ?? {}} onChange={onChange} />
      </motion.div>
      <motion.div className="glass section" {...rise(0.12)}>
        <h3>Audio</h3>
        <Volume id="vol-music" label="Music" value={s.musicVolume} onCommit={(v) => onChange({ musicVolume: v })} />
        <Volume id="vol-sfx" label="Sound effects" value={s.sfxVolume} onCommit={(v) => onChange({ sfxVolume: v })} />
      </motion.div>
      <motion.div className="glass section" {...rise(0.21)}>
        <h3>Saves</h3>
        {game && (
          <div className="srow">
            <div className="slabel">
              <b>Saved in</b>
              <span className="path">{copySavesPath(state.savesFolder, game.discId)}</span>
            </div>
          </div>
        )}
        <ImportSaves gameId={game?.supported ? game.id : undefined} />
      </motion.div>
      <motion.div className="glass section" {...rise(0.24)}>
        <h3>Your copy</h3>
        <div className="srow">
          <div className="slabel"><b>Your copy</b><span>{game ? game.isoPath : 'No copy added yet'}</span></div>
          <div className="row-wrap" style={{ gap: 10 }}>
            <button className="ghost mid" onClick={onChangeCopy}>{game ? 'Change' : 'Add my copy'}</button>
            {game && <button className="ghost mid danger" onClick={() => onRemoveCopy(game.discId)}>Remove</button>}
          </div>
        </div>
        <div className="srow">
          <div className="slabel">
            <b>Decrypted executable (developer build)</b>
            <span>{state.executable.found ? state.executable.path : 'Not found'}</span>
          </div>
          <button className="ghost mid" onClick={onLocateExecutable}>Locate</button>
        </div>
      </motion.div>
    </section>
  );
}
