import java.util.HashMap;
import java.util.Map;
import java.util.prefs.AbstractPreferences;
import java.util.prefs.Preferences;
import java.util.prefs.PreferencesFactory;
/** Process-local preferences: no macOS preference writes or filesystem storage. */
public final class DriveMemoryPreferencesFactory implements PreferencesFactory {
    private static final Preferences USER = new MemoryNode(null, "");
    private static final Preferences SYSTEM = new MemoryNode(null, "");
    public Preferences userRoot() { return USER; }
    public Preferences systemRoot() { return SYSTEM; }
    private static final class MemoryNode extends AbstractPreferences {
        private final Map<String,String> values = new HashMap<>();
        private final Map<String,MemoryNode> nodes = new HashMap<>();
        MemoryNode(AbstractPreferences parent, String name) { super(parent,name); }
        protected void putSpi(String key,String value) { values.put(key,value); }
        protected String getSpi(String key) { return values.get(key); }
        protected void removeSpi(String key) { values.remove(key); }
        protected void removeNodeSpi() { values.clear(); nodes.clear(); }
        protected String[] keysSpi() { return values.keySet().toArray(new String[0]); }
        protected String[] childrenNamesSpi() { return nodes.keySet().toArray(new String[0]); }
        protected AbstractPreferences childSpi(String name) {
            return nodes.computeIfAbsent(name,n -> new MemoryNode(this,n));
        }
        protected void syncSpi() {}
        protected void flushSpi() {}
    }
}
