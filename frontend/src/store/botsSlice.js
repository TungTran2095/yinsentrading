import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { toast } from 'react-toastify';
import api from '../services/api';

// Async thunks
export const fetchBots = createAsyncThunk(
  'bots/fetchBots',
  async (_, { rejectWithValue }) => {
    try {
      const response = await api.get('/bots');
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || { message: error.message });
    }
  }
);

export const fetchBotById = createAsyncThunk(
  'bots/fetchBotById',
  async (id, { rejectWithValue }) => {
    try {
      const response = await api.get(`/bots/${id}`);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || { message: error.message });
    }
  }
);

export const createBot = createAsyncThunk(
  'bots/createBot',
  async (botConfig, { rejectWithValue }) => {
    try {
      const response = await api.post('/bots', { config: botConfig });
      toast.success('Bot created successfully');
      return response.data;
    } catch (error) {
      toast.error(`Failed to create bot: ${error.response?.data?.detail || error.message}`);
      return rejectWithValue(error.response?.data || { message: error.message });
    }
  }
);

export const updateBot = createAsyncThunk(
  'bots/updateBot',
  async ({ id, config }, { rejectWithValue }) => {
    try {
      const response = await api.put(`/bots/${id}`, { config });
      toast.success('Bot updated successfully');
      return response.data;
    } catch (error) {
      toast.error(`Failed to update bot: ${error.response?.data?.detail || error.message}`);
      return rejectWithValue(error.response?.data || { message: error.message });
    }
  }
);

export const startBot = createAsyncThunk(
  'bots/startBot',
  async (id, { rejectWithValue }) => {
    try {
      const response = await api.post(`/bots/${id}/start`);
      toast.success('Bot started successfully');
      return { id, status: 'running' };
    } catch (error) {
      toast.error(`Failed to start bot: ${error.response?.data?.detail || error.message}`);
      return rejectWithValue(error.response?.data || { message: error.message });
    }
  }
);

export const stopBot = createAsyncThunk(
  'bots/stopBot',
  async (id, { rejectWithValue }) => {
    try {
      const response = await api.post(`/bots/${id}/stop`);
      toast.success('Bot stopped successfully');
      return { id, status: 'stopped' };
    } catch (error) {
      toast.error(`Failed to stop bot: ${error.response?.data?.detail || error.message}`);
      return rejectWithValue(error.response?.data || { message: error.message });
    }
  }
);

export const deleteBot = createAsyncThunk(
  'bots/deleteBot',
  async (id, { rejectWithValue }) => {
    try {
      await api.delete(`/bots/${id}`);
      toast.success('Bot deleted successfully');
      return id;
    } catch (error) {
      toast.error(`Failed to delete bot: ${error.response?.data?.detail || error.message}`);
      return rejectWithValue(error.response?.data || { message: error.message });
    }
  }
);

// Slice
const botsSlice = createSlice({
  name: 'bots',
  initialState: {
    bots: [],
    currentBot: null,
    loading: false,
    error: null,
  },
  reducers: {
    clearCurrentBot: (state) => {
      state.currentBot = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // fetchBots
      .addCase(fetchBots.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchBots.fulfilled, (state, action) => {
        state.bots = action.payload;
        state.loading = false;
      })
      .addCase(fetchBots.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || { message: 'Failed to fetch bots' };
      })
      
      // fetchBotById
      .addCase(fetchBotById.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchBotById.fulfilled, (state, action) => {
        state.currentBot = action.payload;
        state.loading = false;
      })
      .addCase(fetchBotById.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || { message: 'Failed to fetch bot' };
      })
      
      // createBot
      .addCase(createBot.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(createBot.fulfilled, (state, action) => {
        state.bots.push(action.payload);
        state.currentBot = action.payload;
        state.loading = false;
      })
      .addCase(createBot.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || { message: 'Failed to create bot' };
      })
      
      // updateBot
      .addCase(updateBot.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(updateBot.fulfilled, (state, action) => {
        const index = state.bots.findIndex(bot => bot.id === action.payload.id);
        if (index !== -1) {
          state.bots[index] = action.payload;
        }
        state.currentBot = action.payload;
        state.loading = false;
      })
      .addCase(updateBot.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || { message: 'Failed to update bot' };
      })
      
      // startBot
      .addCase(startBot.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(startBot.fulfilled, (state, action) => {
        const index = state.bots.findIndex(bot => bot.id === action.payload.id);
        if (index !== -1) {
          state.bots[index].status = action.payload.status;
        }
        if (state.currentBot && state.currentBot.id === action.payload.id) {
          state.currentBot.status = action.payload.status;
        }
        state.loading = false;
      })
      .addCase(startBot.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || { message: 'Failed to start bot' };
      })
      
      // stopBot
      .addCase(stopBot.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(stopBot.fulfilled, (state, action) => {
        const index = state.bots.findIndex(bot => bot.id === action.payload.id);
        if (index !== -1) {
          state.bots[index].status = action.payload.status;
        }
        if (state.currentBot && state.currentBot.id === action.payload.id) {
          state.currentBot.status = action.payload.status;
        }
        state.loading = false;
      })
      .addCase(stopBot.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || { message: 'Failed to stop bot' };
      })
      
      // deleteBot
      .addCase(deleteBot.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(deleteBot.fulfilled, (state, action) => {
        state.bots = state.bots.filter(bot => bot.id !== action.payload);
        if (state.currentBot && state.currentBot.id === action.payload) {
          state.currentBot = null;
        }
        state.loading = false;
      })
      .addCase(deleteBot.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload || { message: 'Failed to delete bot' };
      });
  },
});

export const { clearCurrentBot } = botsSlice.actions;

export default botsSlice.reducer;
