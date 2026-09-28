const $ = (sel) => document.querySelector(sel);

const searchInput = $("#search-input");
const resultsEl = $("#results");
const pageTitle = $("#page-title");
const pageSub = $("#page-sub");
const errorEl = $("#error");
const npArt = $("#np-art");
const npTitle = $("#np-title");
const npArtist = $("#np-artist");
const npBadge = $("#np-badge");
const btnLike = $("#btn-like");
const iconHeartO = $("#icon-heart-o");
const iconHeartF = $("#icon-heart-f");
const btnPlay = $("#btn-play");
const iconPlay = $("#icon-play");
const iconPause = $("#icon-pause");
const btnPrev = $("#btn-prev");
const btnNext = $("#btn-next");
const btnShuffle = $("#btn-shuffle");
const btnRepeat = $("#btn-repeat");
const btnVol = $("#btn-vol");
const btnLyrics = $("#btn-lyrics");
const seekEl = $("#seek");
const timeCur = $("#time-cur");
const timeDur = $("#time-dur");
const volumeEl = $("#volume");
const navLiked = $("#nav-liked");
const likedCount = $("#liked-count");
const playlistItems = $("#playlist-items");
const btnNewPlaylist = $("#btn-new-playlist");
const lyricsView = $("#lyrics-view");
const lyricsBg = $("#lyrics-bg");
const lyricsTitle = $("#lyrics-title");
const lyricsArtist = $("#lyrics-artist");
const lyricsClose = $("#lyrics-close");
const lyricsEmpty = $("#lyrics-empty");
const amLyrics = $("#am-lyrics");

const toastEl = document.createElement("div");
toastEl.className = "toast";
document.body.appendChild(toastEl);

const PLAY_ICON =
  '<svg viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 3 20 12 6 21 6 3"/></svg>';
const NOTE_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>';
const DOTS_ICON =
  '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.8"/><circle cx="12" cy="12" r="1.8"/><circle cx="19" cy="12" r="1.8"/></svg>';
const HEART_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>';
const HEART_ICON_FILL =
  '<svg viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>';
const PLUS_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>';
const LIST_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>';
const TRASH_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';
const PENCIL_ICON =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.83 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>';

const DEFAULT_QUERY = "trending hindi songs";
const LIB_KEY = "aria.library.v1";

const audio = new Audio();
audio.preload = "metadata";
audio.volume = 0.8;

const state = {
  queue: [],
  index: -1,
  current: null,
  shuffle: false,
  repeat: false,
  controller: null,
  view: { type: "search" },
  searchResults: null,
  lastQuery: "",
  lyricsOpen: false,
  lyricsRaf: null,
  lyricsEmptySince: null,
};

let menuEl = null;
let toastTimer = null;
let seeking = false;
let prevVolume = 0.8;

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const fmtTime = (s) => {
  if (!Number.isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
};

const fmtCount = (n) => {
  if (n >= 1e7) return `${(n / 1e7).toFixed(1)}Cr plays`;
  if (n >= 1e5) return `${(n / 1e5).toFixed(1)}L plays`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K plays`;
  return `${n || 0} plays`;
};

const artistNames = (song) =>
  (song.primary_artists && song.primary_artists.join(", ")) || song.subtitle || "";

function loadLibrary() {
  try {
    const data = JSON.parse(localStorage.getItem(LIB_KEY));
    if (data && Array.isArray(data.liked) && Array.isArray(data.playlists)) return data;
  } catch (err) {}
  return { liked: [], playlists: [] };
}

function saveLibrary() {
  try {
    localStorage.setItem(LIB_KEY, JSON.stringify(library));
  } catch (err) {}
}

let library = loadLibrary();

const isLiked = (id) => library.liked.some((s) => s.id === id);
const getPlaylist = (id) => library.playlists.find((p) => p.id === id);

function toggleLike(song) {
  if (isLiked(song.id)) {
    library.liked = library.liked.filter((s) => s.id !== song.id);
    toast("Removed from Liked Songs");
  } else {
    library.liked.unshift(song);
    toast("Added to Liked Songs");
  }
  saveLibrary();
  renderSidebar();
  syncLikeButtons();
  if (state.view.type === "liked") renderMain();
}

function createPlaylist(name) {
  const pl = { id: "p" + Date.now().toString(36), name, songs: [] };
  library.playlists.push(pl);
  saveLibrary();
  renderSidebar();
  return pl;
}

function addToPlaylist(pid, song) {
  const pl = getPlaylist(pid);
  if (!pl) return;
  if (pl.songs.some((s) => s.id === song.id)) {
    toast(`Already in ${pl.name}`);
    return;
  }
  pl.songs.push(song);
  saveLibrary();
  renderSidebar();
  if (state.view.type === "playlist" && state.view.id === pid) renderMain();
  toast(`Added to ${pl.name}`);
}

function removeFromPlaylist(pid, songId) {
  const pl = getPlaylist(pid);
  if (!pl) return;
  pl.songs = pl.songs.filter((s) => s.id !== songId);
  saveLibrary();
  renderSidebar();
  renderMain();
  toast("Removed from playlist");
}

function renamePlaylist(pid, name) {
  const pl = getPlaylist(pid);
  if (!pl) return;
  pl.name = name;
  saveLibrary();
  renderSidebar();
  renderMain();
}

function deletePlaylist(pid) {
  const pl = getPlaylist(pid);
  if (!pl) return;
  library.playlists = library.playlists.filter((p) => p.id !== pid);
  saveLibrary();
  toast(`Deleted "${pl.name}"`);
  go({ type: "search" });
}

function toast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 2200);
}

function closeMenu() {
  if (menuEl) {
    menuEl.remove();
    menuEl = null;
  }
}

function showMenu(rect, items) {
  closeMenu();
  menuEl = document.createElement("div");
  menuEl.className = "menu";
  items.forEach((it) => {
    if (it.sep) {
      const sep = document.createElement("div");
      sep.className = "menu-sep";
      menuEl.appendChild(sep);
      return;
    }
    if (it.header) {
      const h = document.createElement("div");
      h.className = "menu-label";
      h.textContent = it.header;
      menuEl.appendChild(h);
      return;
    }
    if (it.note) {
      const n = document.createElement("div");
      n.className = "menu-empty";
      n.textContent = it.note;
      menuEl.appendChild(n);
      return;
    }
    const btn = document.createElement("button");
    btn.className = "menu-item" + (it.danger ? " danger" : "");
    btn.innerHTML = `${it.icon || ""}<span>${esc(it.label)}</span>`;
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      it.action();
    });
    menuEl.appendChild(btn);
  });
  document.body.appendChild(menuEl);
  const mw = menuEl.offsetWidth;
  const mh = menuEl.offsetHeight;
  let left = Math.min(rect.left, innerWidth - mw - 10);
  let top = rect.bottom + 8;
  if (top + mh > innerHeight - 88) top = rect.top - mh - 8;
  menuEl.style.left = Math.max(10, left) + "px";
  menuEl.style.top = Math.max(10, top) + "px";
}

function openSongMenu(rect, song, ctx, onChange) {
  const liked = isLiked(song.id);
  const items = [
    {
      icon: liked ? HEART_ICON_FILL : HEART_ICON,
      label: liked ? "Unlike" : "Like",
      action: () => {
        toggleLike(song);
        onChange();
      },
    },
    { sep: true },
    {
      icon: PLUS_ICON,
      label: "Add to Playlist",
      action: () => openAddToMenu(rect, song),
    },
  ];
  if (ctx.type === "playlist") {
    items.push(
      { sep: true },
      {
        icon: TRASH_ICON,
        label: "Remove from This Playlist",
        danger: true,
        action: () => {
          removeFromPlaylist(ctx.pid, song.id);
          onChange();
        },
      }
    );
  }
  showMenu(rect, items);
}

function openAddToMenu(rect, song) {
  const items = [{ header: "Add to Playlist" }];
  library.playlists.forEach((pl) =>
    items.push({
      icon: LIST_ICON,
      label: pl.name,
      action: () => {
        addToPlaylist(pl.id, song);
        closeMenu();
      },
    })
  );
  if (!library.playlists.length) items.push({ note: "No playlists yet" });
  items.push(
    { sep: true },
    {
      icon: PLUS_ICON,
      label: "New Playlist…",
      action: () => {
        const name = prompt("Playlist name");
        if (name && name.trim()) {
          const pl = createPlaylist(name.trim());
          addToPlaylist(pl.id, song);
        }
        closeMenu();
      },
    }
  );
  showMenu(rect, items);
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}

function hideError() {
  errorEl.classList.add("hidden");
}

function setFill(input) {
  const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
  input.style.setProperty("--fill", `${pct}%`);
}

function renderSkeleton() {
  let cards = "";
  for (let i = 0; i < 12; i++) {
    cards += `<div class="skel-card"><div class="skel skel-art"></div><div class="skel skel-line"></div><div class="skel skel-line short"></div></div>`;
  }
  resultsEl.innerHTML = `<div class="grid">${cards}</div>`;
}

function artBlock(song) {
  const img = song.image
    ? `<img src="${esc(song.image)}" alt="" loading="lazy" onerror="this.remove()">`
    : "";
  return `<div class="art-note">${NOTE_ICON}</div>${img}`;
}

function cardEl(song, index, list, ctx) {
  const el = document.createElement("article");
  el.className = "card";
  el.dataset.index = index;
  el.innerHTML = `
    <div class="art">
      ${artBlock(song)}
      <div class="eq"><span></span><span></span><span></span></div>
      <div class="overlay"><button class="ov-play" aria-label="Play">${PLAY_ICON}</button></div>
      <button class="card-dots" aria-label="More options">${DOTS_ICON}</button>
    </div>
    <h3>${esc(song.title)}</h3>
    <div class="artist">${esc(artistNames(song))}</div>`;
  el.addEventListener("click", () => playFrom(list, index));
  el.querySelector(".card-dots").addEventListener("click", (e) => {
    e.stopPropagation();
    openSongMenu(e.currentTarget.getBoundingClientRect(), song, ctx, renderMain);
  });
  return el;
}

function heroEl(song, index, list) {
  const el = document.createElement("section");
  el.className = "hero";
  el.dataset.index = index;
  const bits = [song.language, song.year, fmtCount(song.play_count)].filter(Boolean).join(" · ");
  el.innerHTML = `
    <div class="hero-art">
      ${artBlock(song)}
      <div class="eq"><span></span><span></span><span></span></div>
      <div class="overlay"><button class="ov-play" aria-label="Play">${PLAY_ICON}</button></div>
      <button class="card-dots" aria-label="More options">${DOTS_ICON}</button>
    </div>
    <div class="hero-info">
      <div class="kicker">Top Result</div>
      <h2>${esc(song.title)}</h2>
      <div class="artist">${esc(artistNames(song))}</div>
      <div class="meta">${esc(bits)}</div>
    </div>`;
  el.addEventListener("click", () => playFrom(list, index));
  el.querySelector(".card-dots").addEventListener("click", (e) => {
    e.stopPropagation();
    openSongMenu(e.currentTarget.getBoundingClientRect(), song, { type: "search" }, renderMain);
  });
  return el;
}

function renderResultsFrom(list) {
  resultsEl.innerHTML = "";
  if (!list.length) {
    resultsEl.innerHTML = `<div class="empty">${NOTE_ICON}<p>No results found</p></div>`;
    return;
  }
  resultsEl.appendChild(heroEl(list[0], 0, list));
  if (list.length > 1) {
    const h = document.createElement("h2");
    h.className = "section-title";
    h.textContent = "Songs";
    resultsEl.appendChild(h);
    const grid = document.createElement("div");
    grid.className = "grid";
    for (let i = 1; i < list.length; i++) grid.appendChild(cardEl(list[i], i, list, { type: "search" }));
    resultsEl.appendChild(grid);
  }
  syncCardStates();
}

function renderCollection(kicker, name, songs, ctx, pl) {
  pageTitle.textContent = name;
  pageSub.textContent = "";
  resultsEl.innerHTML = "";
  const totalSec = songs.reduce((a, s) => a + (s.duration_seconds || 0), 0);
  const mins = totalSec ? Math.round(totalSec / 60) : 0;
  const head = document.createElement("div");
  head.className = "coll-head";
  const cover = songs[0] && songs[0].image
    ? `<img src="${esc(songs[0].image)}" alt="" onerror="this.remove()">${NOTE_ICON}`
    : ctx.type === "liked"
      ? HEART_ICON_FILL
      : NOTE_ICON;
  head.innerHTML = `
    <div class="coll-cover">${cover}</div>
    <div class="coll-info">
      <div class="kicker">${esc(kicker)}</div>
      <h2>${esc(name)}</h2>
      <div class="coll-meta">${songs.length} ${songs.length === 1 ? "song" : "songs"}${mins ? ` · ${mins} min` : ""}</div>
    </div>`;
  const actions = document.createElement("div");
  actions.className = "coll-actions";
  const play = document.createElement("button");
  play.className = "coll-play";
  play.setAttribute("aria-label", "Play all");
  play.innerHTML = PLAY_ICON;
  play.addEventListener("click", (e) => {
    e.stopPropagation();
    playFrom(songs, 0);
  });
  actions.appendChild(play);
  if (pl) {
    const ren = document.createElement("button");
    ren.className = "t-btn";
    ren.setAttribute("aria-label", "Rename playlist");
    ren.innerHTML = PENCIL_ICON;
    ren.addEventListener("click", () => {
      const name2 = prompt("Rename playlist", pl.name);
      if (name2 && name2.trim()) renamePlaylist(pl.id, name2.trim());
    });
    const del = document.createElement("button");
    del.className = "t-btn danger";
    del.setAttribute("aria-label", "Delete playlist");
    del.innerHTML = TRASH_ICON;
    del.addEventListener("click", () => {
      if (confirm(`Delete playlist "${pl.name}"?`)) deletePlaylist(pl.id);
    });
    actions.appendChild(ren);
    actions.appendChild(del);
  }
  head.querySelector(".coll-info").appendChild(actions);
  resultsEl.appendChild(head);
  if (!songs.length) {
    const msg =
      ctx.type === "liked"
        ? "Songs you like will appear here"
        : "This playlist is empty";
    const hint =
      ctx.type === "liked"
        ? "Tap the ⋯ menu on any artwork and choose Like"
        : "Find a song, then tap ⋯ on its artwork to add it";
    resultsEl.innerHTML += `<div class="empty">${ctx.type === "liked" ? HEART_ICON_FILL : NOTE_ICON}<p>${msg}</p><p class="hint">${hint}</p></div>`;
    return;
  }
  const grid = document.createElement("div");
  grid.className = "grid";
  songs.forEach((s, i) => grid.appendChild(cardEl(s, i, songs, ctx)));
  resultsEl.appendChild(grid);
  syncCardStates();
}

function renderSearchView() {
  pageTitle.textContent = state.lastQuery || "Search";
  if (state.searchResults) {
    pageSub.textContent = `${state.searchResults.length} songs`;
    renderResultsFrom(state.searchResults);
  } else {
    pageSub.textContent = "Find songs, artists and albums";
    resultsEl.innerHTML = `<div class="empty">${NOTE_ICON}<p>Search for songs, artists and albums</p></div>`;
  }
}

function renderMain() {
  closeMenu();
  const v = state.view;
  if (v.type === "liked") {
    renderCollection("Library", "Liked Songs", library.liked, { type: "liked" });
    return;
  }
  if (v.type === "playlist") {
    const pl = getPlaylist(v.id);
    if (!pl) {
      state.view = { type: "search" };
      renderMain();
      return;
    }
    renderCollection("Playlist", pl.name, pl.songs, { type: "playlist", pid: pl.id }, pl);
    return;
  }
  renderSearchView();
}

function go(view) {
  state.view = view;
  hideError();
  renderSidebar();
  renderMain();
  document.querySelector(".main").scrollTop = 0;
}

function renderSidebar() {
  navLiked.classList.toggle("active", state.view.type === "liked");
  likedCount.textContent = library.liked.length || "";
  if (!library.playlists.length) {
    playlistItems.innerHTML = `<div class="sidebar-empty">No playlists yet</div>`;
    return;
  }
  playlistItems.innerHTML = "";
  library.playlists.forEach((pl) => {
    const btn = document.createElement("button");
    btn.className =
      "nav-item" + (state.view.type === "playlist" && state.view.id === pl.id ? " active" : "");
    btn.innerHTML = `${LIST_ICON}<span class="nav-label">${esc(pl.name)}</span><span class="nav-count">${pl.songs.length || ""}</span>`;
    btn.addEventListener("click", () => go({ type: "playlist", id: pl.id }));
    playlistItems.appendChild(btn);
  });
}

function playFrom(list, index) {
  state.queue = list;
  playIndex(index);
}

function playIndex(i) {
  const song = state.queue[i];
  if (!song || !song.media_url) return;
  state.index = i;
  state.current = song;
  audio.src = song.media_url;
  audio.play().catch(() => showError("Playback failed — try another song"));
  updateNowPlaying();
  syncLikeButtons();
  syncCardStates();
  updateMediaSession(song);
  hideError();
  if (state.lyricsOpen) openLyrics();
}

function togglePlay() {
  if (!audio.src) {
    if (state.queue.length) playIndex(0);
    return;
  }
  if (audio.paused) audio.play().catch(() => {});
  else audio.pause();
}

function nextTrack(auto) {
  const n = state.queue.length;
  if (!n) return;
  let i;
  if (state.shuffle && n > 1) {
    do {
      i = Math.floor(Math.random() * n);
    } while (i === state.index);
  } else {
    i = state.index + 1;
    if (i >= n) {
      if (state.repeat || !auto) i = 0;
      else {
        audio.pause();
        return;
      }
    }
  }
  playIndex(i);
}

function prevTrack() {
  if (audio.currentTime > 3) {
    audio.currentTime = 0;
    return;
  }
  const n = state.queue.length;
  if (!n) return;
  playIndex(state.index > 0 ? state.index - 1 : n - 1);
}

function syncPlayState() {
  iconPlay.classList.toggle("hidden", !audio.paused);
  iconPause.classList.toggle("hidden", audio.paused);
  btnPlay.setAttribute("aria-label", audio.paused ? "Play" : "Pause");
  if ("mediaSession" in navigator) {
    navigator.mediaSession.playbackState = audio.paused ? "paused" : "playing";
  }
  syncCardStates();
}

function syncLikeButtons() {
  const liked = !!(state.current && isLiked(state.current.id));
  btnLike.classList.toggle("on", liked);
  iconHeartO.classList.toggle("hidden", liked);
  iconHeartF.classList.toggle("hidden", !liked);
}

function syncCardStates() {
  document.querySelectorAll("[data-index]").forEach((el) => {
    el.classList.toggle("playing", Number(el.dataset.index) === state.index && !!state.current);
  });
}

function updateNowPlaying() {
  const song = state.current;
  if (!song) return;
  npArt.style.display = song.image ? "block" : "none";
  npArt.src = song.image || "";
  npTitle.textContent = song.title;
  npArtist.textContent = artistNames(song);
  npBadge.classList.toggle("hidden", song.media_quality !== "320kbps");
}

function updateMediaSession(song) {
  if (!("mediaSession" in navigator)) return;
  navigator.mediaSession.metadata = new MediaMetadata({
    title: song.title,
    artist: artistNames(song),
    album: song.album || "",
    artwork: song.image ? [{ src: song.image, sizes: "500x500", type: "image/jpeg" }] : [],
  });
  navigator.mediaSession.setActionHandler("play", togglePlay);
  navigator.mediaSession.setActionHandler("pause", togglePlay);
  navigator.mediaSession.setActionHandler("previoustrack", prevTrack);
  navigator.mediaSession.setActionHandler("nexttrack", () => nextTrack(false));
}

const amReady =
  "customElements" in window
    ? customElements.whenDefined("am-lyrics").catch(() => {})
    : Promise.reject(new Error("no custom elements"));

function openLyrics() {
  const song = state.current;
  if (!song) {
    toast("Play a song first");
    return;
  }
  state.lyricsOpen = true;
  state.lyricsEmptySince = null;
  lyricsEmpty.classList.add("hidden");
  amLyrics.classList.remove("empty");
  lyricsView.classList.add("open");
  lyricsView.setAttribute("aria-hidden", "false");
  if (song.image) lyricsBg.style.backgroundImage = `url("${song.image}")`;
  else lyricsBg.style.backgroundImage = "";
  lyricsTitle.textContent = song.title;
  lyricsArtist.textContent = artistNames(song);
  amReady.then(() => {
    amLyrics.songTitle = song.title;
    amLyrics.songArtist = artistNames(song);
    amLyrics.songAlbum = song.album || "";
    amLyrics.query = `${song.title} ${artistNames(song)}`;
    amLyrics.songDuration = Math.round((song.duration_seconds || 0) * 1000);
    amLyrics.duration = audio.duration
      ? Math.round(audio.duration * 1000)
      : Math.round((song.duration_seconds || 0) * 1000);
    amLyrics.fontFamily = getComputedStyle(document.body).fontFamily;
    amLyrics.hideSourceFooter = true;
    amLyrics.currentTime = audio.currentTime * 1000;
  });
  if (!state.lyricsRaf) loopLyrics();
}

function loopLyrics() {
  state.lyricsRaf = requestAnimationFrame(() => {
    if (!state.lyricsOpen) {
      state.lyricsRaf = null;
      return;
    }
    amLyrics.currentTime = audio.currentTime * 1000;
    const drained =
      !amLyrics.isLoading && (!amLyrics.lyrics || !amLyrics.lyrics.length);
    if (drained) {
      if (!state.lyricsEmptySince) state.lyricsEmptySince = performance.now();
      if (performance.now() - state.lyricsEmptySince > 700) {
        lyricsEmpty.classList.remove("hidden");
        amLyrics.classList.add("empty");
      }
    } else {
      state.lyricsEmptySince = null;
      lyricsEmpty.classList.add("hidden");
      amLyrics.classList.remove("empty");
    }
    loopLyrics();
  });
}

function closeLyrics() {
  state.lyricsOpen = false;
  state.lyricsEmptySince = null;
  lyricsEmpty.classList.add("hidden");
  amLyrics.classList.remove("empty");
  lyricsView.classList.remove("open");
  lyricsView.setAttribute("aria-hidden", "true");
  if (state.lyricsRaf) {
    cancelAnimationFrame(state.lyricsRaf);
    state.lyricsRaf = null;
  }
}

async function runSearch(q) {
  state.lastQuery = q;
  if (state.controller) state.controller.abort();
  state.controller = new AbortController();
  state.view = { type: "search" };
  renderSidebar();
  pageTitle.textContent = q;
  pageSub.textContent = "Searching…";
  hideError();
  renderSkeleton();
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=36`, {
      signal: state.controller.signal,
    });
    const data = await res.json();
    if (!res.ok || data.status !== "success") {
      throw new Error(data.message || "Search failed");
    }
    state.searchResults = data.results;
    if (state.current) {
      state.index = state.searchResults.findIndex((s) => s.id === state.current.id);
    }
    if (state.view.type === "search") renderSearchView();
  } catch (err) {
    if (err.name === "AbortError") return;
    if (state.view.type !== "search") return;
    renderSearchView();
    showError(err.message || "Something went wrong");
  }
}

searchInput.addEventListener("input", () => {
  clearTimeout(searchInput._t);
  const q = searchInput.value.trim();
  if (!q) return;
  searchInput._t = setTimeout(() => runSearch(q), 450);
});

searchInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  clearTimeout(searchInput._t);
  const q = searchInput.value.trim();
  if (q) runSearch(q);
});

navLiked.addEventListener("click", () => go({ type: "liked" }));

btnNewPlaylist.addEventListener("click", () => {
  const name = prompt("Playlist name");
  if (!name || !name.trim()) return;
  const pl = createPlaylist(name.trim());
  go({ type: "playlist", id: pl.id });
});

btnPlay.addEventListener("click", togglePlay);
btnNext.addEventListener("click", () => nextTrack(false));
btnPrev.addEventListener("click", prevTrack);

btnLike.addEventListener("click", () => {
  if (!state.current) {
    toast("Play a song first");
    return;
  }
  toggleLike(state.current);
  if (state.view.type === "liked" || state.view.type === "playlist") renderMain();
});

btnShuffle.addEventListener("click", () => {
  state.shuffle = !state.shuffle;
  btnShuffle.classList.toggle("on", state.shuffle);
});

btnRepeat.addEventListener("click", () => {
  state.repeat = !state.repeat;
  btnRepeat.classList.toggle("on", state.repeat);
});

btnLyrics.addEventListener("click", () => {
  if (state.lyricsOpen) closeLyrics();
  else openLyrics();
});

lyricsClose.addEventListener("click", closeLyrics);

amLyrics.addEventListener("line-click", (e) => {
  const ts = e.detail && e.detail.timestamp;
  if (typeof ts === "number" && Number.isFinite(ts)) {
    audio.currentTime = ts / 1000;
    if (audio.paused) audio.play().catch(() => {});
  }
});

btnVol.addEventListener("click", () => {
  if (audio.volume > 0) {
    prevVolume = audio.volume;
    audio.volume = 0;
    volumeEl.value = 0;
  } else {
    audio.volume = prevVolume;
    volumeEl.value = prevVolume * 100;
  }
  setFill(volumeEl);
});

volumeEl.addEventListener("input", () => {
  audio.volume = volumeEl.value / 100;
  setFill(volumeEl);
});

seekEl.addEventListener("input", () => {
  seeking = true;
  setFill(seekEl);
  if (audio.duration) timeCur.textContent = fmtTime((seekEl.value / 100) * audio.duration);
});

seekEl.addEventListener("change", () => {
  if (audio.duration) audio.currentTime = (seekEl.value / 100) * audio.duration;
  seeking = false;
});

audio.addEventListener("timeupdate", () => {
  if (seeking || !audio.duration) return;
  const pct = (audio.currentTime / audio.duration) * 100;
  seekEl.value = pct;
  setFill(seekEl);
  timeCur.textContent = fmtTime(audio.currentTime);
});

audio.addEventListener("loadedmetadata", () => {
  timeDur.textContent = fmtTime(audio.duration);
});

audio.addEventListener("play", syncPlayState);
audio.addEventListener("pause", syncPlayState);

audio.addEventListener("ended", () => {
  if (state.repeat && !state.shuffle) {
    playIndex(0);
    return;
  }
  nextTrack(true);
});

audio.addEventListener("error", () => {
  if (audio.src && state.current) showError(`Could not stream "${state.current.title}"`);
});

document.addEventListener("click", (e) => {
  if (menuEl && !menuEl.contains(e.target)) closeMenu();
});

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") {
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (e.key === "Escape") {
    if (menuEl) closeMenu();
    else if (state.lyricsOpen) closeLyrics();
    return;
  }
  if (e.code === "Space") {
    e.preventDefault();
    togglePlay();
  } else if (e.key === "ArrowRight" && audio.duration) {
    audio.currentTime = Math.min(audio.currentTime + 10, audio.duration);
  } else if (e.key === "ArrowLeft" && audio.duration) {
    audio.currentTime = Math.max(audio.currentTime - 10, 0);
  } else if (e.key === "l" || e.key === "L") {
    if (state.lyricsOpen) closeLyrics();
    else openLyrics();
  } else if (e.key === "/") {
    e.preventDefault();
    searchInput.focus();
  }
});

setFill(volumeEl);
setFill(seekEl);
renderSidebar();
runSearch(DEFAULT_QUERY);
